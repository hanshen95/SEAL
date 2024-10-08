import argparse
import math
import os
from datetime import datetime
import torch
import jsonlines
from transformers.trainer import get_scheduler

from seal.datasets import SFTDataset, SFTDatasetIndexed
from seal.models import Actor, TrainableTensorModule
from seal.trainer import SFTTrainer, SFTSelectorTrainer
from seal.utils import blending_datasets, get_strategy, get_tokenizer


def train(args):
    
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()

    # configure model
    # load huggingface model
    model = Actor(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        ds_config=strategy.get_ds_train_config(is_actor=True),
        gradient_checkpointing=args.gradient_checkpointing,
    )

    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, model.model, "right", strategy, use_fast=not args.disable_fast_tokenizer)

    strategy.print(model)
    
    # load weights for ref model
    if args.ref_constant:
        ref_model = Actor(
            args.pretrain,
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            ds_config=strategy.get_ds_eval_config(offload=args.ref_offload),
        )
        if args.ref_offload:
            ref_model._offload = True
        get_tokenizer(args.pretrain, ref_model.model, "right", strategy, use_fast=not args.disable_fast_tokenizer)
        print("\n TODO: implement ref regularization. Right now there will be no reg.\n")
    else:
        ref_model=None

    # configure optimizer
    optim = strategy.create_optimizer(model, lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.l2)

    # prepare for data and dataset
    train_data, eval_data = blending_datasets(
    args.dataset, args.dataset_probs, strategy, args.seed, max_count=args.max_samples
)
    new_data = blending_datasets(
        args.new_dataset,
        "1",
        strategy,
        args.seed,
        return_eval=False,
        max_count=args.max_samples,
    )
    train_data = train_data.select(range(min(args.max_samples, len(train_data))))
    eval_data = eval_data.select(range(min(args.max_samples, len(eval_data))))
    new_data = new_data.select(range(min(args.max_samples, len(new_data))))
    train_dataset = SFTDataset(
        train_data,
        tokenizer,
        args.max_len,
        strategy,
        pretrain_mode=args.pretrain_mode,
        input_template=args.input_template,
    )
    eval_dataset = SFTDataset(
        eval_data,
        tokenizer,
        args.max_len,
        strategy,
        pretrain_mode=args.pretrain_mode,
        input_template=args.input_template,
    )
    new_dataset = SFTDatasetIndexed(
        new_data,
        tokenizer,
        args.max_len,
        strategy,
        pretrain_mode=args.pretrain_mode,
        input_template=args.input_template,
    )
    
    # initilize data selector
    p = TrainableTensorModule(size=new_dataset.__len__(),activation=args.selector_activation)
    p_opt=strategy.create_optimizer(p, lr=args.selector_learning_rate, betas=(0.9, 0.95), weight_decay=0.)
    

    train_dataloader = strategy.setup_dataloader(
        train_dataset, args.micro_train_batch_size, True, True, train_dataset.collate_fn
    )
    eval_dataloader = strategy.setup_dataloader(
        eval_dataset, args.micro_train_batch_size, True, False, eval_dataset.collate_fn
    )
    new_dataloader = strategy.setup_dataloader(
        new_dataset, args.micro_train_batch_size, True, True, new_dataset.collate_fn
    )

    # scheduler
    num_update_steps_per_epoch = len(train_dataloader) // strategy.accumulated_gradient
    max_steps = math.ceil(args.max_epochs * num_update_steps_per_epoch)

    scheduler = get_scheduler(
        args.lr_scheduler,
        optim,
        num_warmup_steps=math.ceil(max_steps * 0.03),
        num_training_steps=max_steps,
    )
    p_scheduler = get_scheduler(
        args.selector_lr_scheduler,
        p_opt,
        num_warmup_steps=math.ceil(max_steps * 0.03),
        num_training_steps=max_steps,
    )

    # gradient_checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )

    # # prepare models
    # (model, optim, scheduler) = strategy.prepare((model, optim, scheduler))
    # strategy prepare
    if ref_model:
        ((model, optim, scheduler),(p,p_opt,p_scheduler),ref_model) = strategy.prepare((model, optim, scheduler),(p,p_opt,p_scheduler),ref_model)
    else:
        ((model, optim, scheduler),(p,p_opt,p_scheduler)) = strategy.prepare((model, optim, scheduler),(p,p_opt,p_scheduler))

    # load checkpoint
    if args.load_checkpoint:
        strategy.print("Load checkpoint: ", args.save_path)
    if args.save_path:
        os.makedirs(args.save_path, exist_ok=True)

    # configure Trainer
    trainer = SFTSelectorTrainer(
        model=model,
        ref_model=ref_model,
        ref_constant=args.ref_constant,
        strategy=strategy,
        optim=optim,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        new_dataloader=new_dataloader,
        p=p,
        p_opt=p_opt,
        p_scheduler=p_scheduler, 
        scheduler=scheduler,
        max_norm=args.max_norm,
        pretrain_mode=args.pretrain_mode,
        batch_size=args.train_batch_size,
        max_epochs=args.max_epochs,
        tokenizer=tokenizer,
    )

    trainer.fit(args)
    
    # save model checkpoint after fitting on only rank0
    if args.save_path:
        strategy.save_model(model, tokenizer, args.save_path)
    p_tensor = trainer.p.logits
    if strategy.is_rank_0():
        print(p_tensor)
        torch.save(p_tensor, "./ckpt/"+args.selector_name+"_"+ \
                   args.selector_activation+".pt")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", type=str, default="bigscience/bloomz-1b7")
    parser.add_argument("--dataset", type=str, default="Dahoas/full-hh-rlhf")
    parser.add_argument("--dataset_probs", type=str, default="1.0", help="sampling probs for datasets")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_sft")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1000)  # 1000GB
    parser.add_argument("--max_epochs", type=int, default=2)
    parser.add_argument("--micro_train_batch_size", type=int, default=8)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--max_samples", type=int, default=1000000)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--l2", type=float, default=0)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--load_checkpoint", action="store_true", default=False)
    parser.add_argument("--pretrain_mode", action="store_true", default=False)

    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--learning_rate", type=float, default=2e-6)
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False)
    parser.add_argument("--flash_attn", action="store_true", default=False)
    parser.add_argument("--aux_loss_coef", type=float, default=0)
    parser.add_argument("--grad_accum_dtype", type=str, default=None)
    parser.add_argument("--disable_trace_cache", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--lora_dropout", type=float, default=0)
    parser.add_argument("--input_template", type=str, default="Human: {}\nAssistant: ")
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true")
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    
    # data selector parameters
    parser.add_argument("--new_dataset", type=str, default=None)
    parser.add_argument("--selector_learning_rate", type=float, default=1e-2)
    parser.add_argument("--selector_activation", type=str, default="softmax")
    parser.add_argument("--selector_name", type=str, default="sft_selection_logits")
    parser.add_argument("--ref_constant", type=float, default=0.)
    parser.add_argument("--upperlevel_weight", type=float, default=0.9)
    parser.add_argument("--upperlevel_weight_decay", type=float, default=0.1)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--selector_lr_scheduler", type=str, default="constant")

    # custom dataset key name
    parser.add_argument("--input_key", type=str, default=None)
    parser.add_argument("--output_key", type=str, default=None)
    
    # wandb pamameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_sft")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="sft_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    args = parser.parse_args()
    torch.cuda.empty_cache()
    train(args)
