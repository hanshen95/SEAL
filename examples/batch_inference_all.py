import argparse
import os
import subprocess
from datetime import timedelta

import jsonlines
import torch
from torch import distributed as dist
from tqdm import tqdm
from transformers import AutoTokenizer

from openrlhf.datasets import PromptDataset

from seal.datasets import SFTDataset
from seal.models import Actor 
from seal.utils import blending_datasets, get_processor, get_strategy, get_tokenizer

        
def batch_generate(args):
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed(timeout=timedelta(minutes=720))

    # configure model
    model = Actor(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
    )

    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, model.model, "left", strategy, use_fast=not args.disable_fast_tokenizer)

    # prepare models
    model = strategy.prepare(model)
    model.eval()

    # tokenizer
    def tokenize_fn(texts):
        batch = tokenizer(
            texts,
            return_tensors="pt",
            max_length=args.prompt_max_len,
            padding=True,
            truncation=True,
        )
        return {k: v.to(torch.cuda.current_device()) for k, v in batch.items()}

    prompts_data = blending_datasets(
        args.dataset,
        args.dataset_probs,
        strategy,
        args.seed,
        return_eval=False,
        max_count=args.max_samples,
    )
    if args.iter is None:
        prompts_data = prompts_data.select(range(min(args.max_samples, len(prompts_data))))
    else:
        # for iterative generation
        start_idx = args.iter * args.rollout_batch_size
        end_idx = start_idx + args.rollout_batch_size
        prompts_data = prompts_data.select(range(start_idx, min(end_idx, len(prompts_data))))

    prompts_dataset = PromptDataset(prompts_data, tokenizer, strategy, input_template=args.input_template)
    prompts_dataloader = strategy.setup_dataloader(
        prompts_dataset, args.micro_batch_size, True, False, drop_last=False
    )
    pbar = tqdm(
        prompts_dataloader,
        disable=not strategy.is_rank_0(),
    )

    dist.barrier()
    N = args.best_of_n
    output_dataset = []
    
    for prompts in pbar:
        # Conditional SFT inference
        if args.enable_ca:
            for i in range(len(prompts)):
                prompts[i] += args.ca_prompt.strip() + " "

        inputs = tokenize_fn(prompts)
        for _ in range(N):
            outputs = model.model.generate(
                **inputs,
                use_cache=True,
                max_length=args.max_len,
                do_sample=not args.greedy_sampling,
                top_p=args.top_p,
                early_stopping=True,
                num_beams=1,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for prompt, output in zip(prompts, outputs):
                output = output[len(prompt) :]
                output_dataset.append({"input": prompt, "output": output})

        dist.barrier()

    with jsonlines.open(args.output_path + str(strategy.get_rank()), mode="w") as writer:
        writer.write_all(output_dataset)

    # wait unitl all processes generate done
    dist.barrier()

    # concate multiple output files in rank 0
    if strategy.is_rank_0():
        output_dataset = []
        world_size = dist.get_world_size()
        files = [args.output_path + str(rank) for rank in range(world_size)]
        for file in files:
            with jsonlines.open(file, mode="r") as reader:
                for obj in reader:
                    output_dataset.append(obj)
            os.remove(file)

        with jsonlines.open(args.output_path, mode="w") as writer:
            writer.write_all(output_dataset)
        if args.convert_for_alpaca_eval:
            # convert jsonlines file to json compatible for alpaca_eval
            command = "sed -i 's/'input'/'instruction'/g' "+args.output_path
            subprocess.call([command],shell=True)
            command = "sed -i '1s/^/[/; $!s/$/,/; $s/$/]/' "+args.output_path
            subprocess.call([command],shell=True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=str, default='./scripts/evaluation_data/')
    parser.add_argument("--convert_for_alpaca_eval", action="store_true", default=False)
    
    parser.add_argument("--eval_task", type=str, default=None, help="set to generate, generate_vllm or rm")
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--max_len", type=int, default=2048)
    parser.add_argument("--zero_stage", type=int, default=0)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--flash_attn", action="store_true", default=False)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--micro_batch_size", type=int, default=16)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dataset_probs", type=str, default="1.0")
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=1000000)
    parser.add_argument("--seed", type=int, default=1234)
    
    # custom dataset key name
    parser.add_argument("--input_key", type=str, default=None)
    parser.add_argument("--output_key", type=str, default=None)

    # for generation
    parser.add_argument("--ta_prompt", type=str, default=None)
    parser.add_argument("--prompt_max_len", type=int, default=1024)
    parser.add_argument("--greedy_sampling", action="store_true", default=False)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--best_of_n", type=int, default=1)
    parser.add_argument("--input_template", type=str, default="Human: {}\nAssistant: ")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument(
        "--post_processor",
        type=str,
        default=None,
        help="set to rs (Rejection Sampling), ca (Conditional SFT) or None",
    )

    # for vllm
    parser.add_argument("--tp_size", type=int, default=8)

    # for Iterative generation and Rejection Sampling
    parser.add_argument("--iter", type=int, default=None)
    parser.add_argument("--rollout_batch_size", type=int, default=2048)

    # for Conditional SFT
    parser.add_argument("--normalize_reward", action="store_true", default=False)
    parser.add_argument("--reward_template", type=str, default=None)
    parser.add_argument("--enable_ca", action="store_true", default=False)
    parser.add_argument("--ca_prompt", type=str, default="<rm_score>: 5.00", help="Conditional SFT prompt")
    
    # to bypass an error #bak
    parser.add_argument("--disable_trace_cache", action="store_true", default=False)
    parser.add_argument("--aux_loss_coef", type=float, default=0)

    args = parser.parse_args()
    
    root = args.output_root
    datasets = ['hh.jsonl','orca.jsonl','category_1.jsonl','category_2.jsonl','category_3.jsonl','category_4.jsonl','category_5.jsonl', \
                'category_6.jsonl', 'category_7.jsonl','category_8.jsonl','category_9.jsonl','category_10.jsonl','category_11.jsonl']
    model_name = args.pretrain.split("/")[-1]
    for dataset in datasets:
        args.dataset = root+dataset
        print("\n Doing batch inference on ", args.dataset)
        args.output_path = root+model_name+'-'+dataset.split(".")[0]+".json"
        if args.eval_task and args.eval_task == "generate":
            batch_generate(args)
        else:
            print("Invalid or missing '--eval_task' argument. Please specify either 'generate' or 'rm'.")
        
        
        
