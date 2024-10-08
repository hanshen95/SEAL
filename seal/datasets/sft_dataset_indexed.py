from typing import Callable
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
from openrlhf.datasets.utils import exist_and_not_none, zero_pad_sequences


def preprocess_data(data, input_template=None, input_key=None, output_key=None):
    # custom dataset
    if input_key and output_key:
        prompt = data[input_key]
        response = data[output_key]
    else:
        # bak
        # Dahoas/full-hh-rlhf
        # iamketan25/open-assistant-instructions
        if exist_and_not_none(data, "prompt") and exist_and_not_none(data, "chosen"):
            prompt = data["prompt"]
            response = data["chosen"]
            input_template = None  # do not modified with input template again
        # pvduy/sharegpt_alpaca_oa_vicuna_format
        elif exist_and_not_none(data, "prompt") and exist_and_not_none(data, "label"):
            prompt = data["prompt"].replace("USER:", "").replace("ASSISTANT:", "")
            response = data["label"].replace("</s>", "")
        # Open-Orca/OpenOrca
        elif exist_and_not_none(data, "system_prompt") and exist_and_not_none(data, "response"):
            # prompt = data["system_prompt"] + " " + data["question"]
            prompt = data["question"] #bak
            response = data["response"]
        # MaziyarPanahi/WizardLM_evol_instruct_V2_196k
        # jondurbin/airoboros-3.2
        elif exist_and_not_none(data, "conversations"):
    
            def process_conversations(lll):
                result = []
                for l in lll:
                    if "human" in l["from"]:
                        result.append(input_template.format(l["value"]))
                    else:
                        result.append(l["value"] + "\n")
                return "".join(result)
    
            prompt = process_conversations(data["conversations"][:-1])
            response = data["conversations"][-1]["value"]
            input_template = None  # do not modified with input template again
        # chargoddard/commitpack-ft-instruct #bak
        # alpaca; open-platypus #bak
        elif exist_and_not_none(data, "instruction") and exist_and_not_none(data,"input") and \
                exist_and_not_none(data,"output"):
            if len(data["input"])>=1:
                prompt = data["instruction"]+'\n'+data["input"]
            else:
                prompt = data["instruction"]
            response = data["output"]
        # for batch_inference.py
        elif exist_and_not_none(data, "input") and exist_and_not_none(data, "output"):
            prompt = data["input"]
            response = data["output"]
            input_template = None
        else:
            raise ValueError("Unknown SFT dataset")

    # input template
    if input_template:
        prompt = input_template.format(prompt)
    return prompt, response


class SFTDatasetIndexed(Dataset):
    """
    Dataset for SFT model

    Args:
        dataset: dataset for SFT model
        tokenizer: tokenizer for SFT model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        input_template="Human: {}\nAssistant: ",
        pretrain_mode=False,
    ) -> None:
        super().__init__()
        self.ids = []
        self.prompts = []
        self.responses = []
        self.prompt_ids_lens = []
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.pretrain_mode = pretrain_mode
        self.max_length = max_length
        input_key = getattr(self.strategy.args, "input_key", None)
        output_key = getattr(self.strategy.args, "output_key", None) 

        k = 0
        for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):
            prompt, response = preprocess_data(data, None if pretrain_mode else input_template, input_key, output_key)

            if not self.pretrain_mode:
                prompt_token = self.tokenizer(
                    prompt,
                    max_length=self.max_length,
                    padding=False,
                    truncation=True,
                    return_tensors="pt",
                )
                prompt_ids_len = prompt_token["attention_mask"].int().sum().item()
            else:
                prompt_ids_len = 0

            if not self.pretrain_mode:
                # filter the sample whose length is greater than max_length (2 for answer length)
                if prompt_ids_len >= self.max_length - 2:
                    continue
                if not prompt or not response:
                    continue

            self.prompt_ids_lens.append(prompt_ids_len)
            self.prompts.append(prompt)
            self.responses.append(response)
            self.ids.append(k)
            k+=1

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        ide = self.ids[idx]
        prompt_ids_len = self.prompt_ids_lens[idx]
        prompt = self.prompts[idx]
        response = self.responses[idx]

        input_token = self.tokenizer(
            prompt + response + " " + self.tokenizer.eos_token,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )
        info = {"input": prompt, "output": response}
        # to avoid EOS_token truncation
        input_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        input_token["attention_mask"][0][-1] = True
        return ide, prompt_ids_len, input_token["input_ids"], input_token["attention_mask"], info

    def collate_fn(self, item_list):
        prompt_ids_lens = []
        input_ids = []
        attention_masks = []
        infos = {"input": [], "output": []}
        ids = []

        for ide, prompt_ids_len, input_id, attention_mask, info in item_list:
            prompt_ids_lens.append(prompt_ids_len)
            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            infos["input"].append(info["input"])
            infos["output"].append(info["output"])
            ids.append(ide)

        input_ids = zero_pad_sequences(input_ids, "right", self.tokenizer.pad_token_id)
        attention_masks = zero_pad_sequences(attention_masks, "right")
        return torch.tensor(ids, dtype=int), prompt_ids_lens, input_ids, attention_masks, infos
