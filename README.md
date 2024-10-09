<div align="center">
    <img alt="SEAL logo" src="./docs/seal_logo.png" style="height: 110px;" />
</div>

<hr>

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/hanshen95/SEAL/blob/main/LICENSE) [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)  [![Arxiv link](https://img.shields.io/badge/cs.LG-queued-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/)



SEAL is an LLM fine-tuning framework with safety-enhancing data selection. This implementation is based on OpenRLHF, DeepSpeed, Transformers and Pytorch.



## Introduction

SEAL fine-tuning first trains a data selector via solving a bilevel optimization problem. Then it filters the fine-tuing dataset with the trained selector by hard-thresholding. Finally we fine-tune the LLM on the filtered dataset. 

<div align="center">
    <img alt="SEAL framework" src="./docs/seal_framework.png" style="height: 280px;" />
</div>
<br/><br/>

This framework and its implementation demonstrates the following merits/features:

- **Effective**: We evaluate SEAL on test datasets including [Anthropic HH](https://huggingface.co/datasets/Anthropic/hh-rlhf), [Slim Orca](https://huggingface.co/datasets/Open-Orca/SlimOrca) and [HEx-PHI](https://huggingface.co/datasets/LLM-Tuning-Safety/HEx-PHI). SEAL consistently outperforms multiple baselines across different models including Llama-3-8b-Instruct, Merlinite-7b and Pythia-2.8b.

- **Flexible and transferable**: The performance is relatively robust to data selection percent, and the trained selector can be transferable between fine tuning different models.

- **Distributed training**: This implementation is based on [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), which uses [DeepSpeed](https://github.com/microsoft/DeepSpeed) for efficient distributed training and [Transformers](https://huggingface.co/docs/transformers/en/index) for easy modification capability.


## Example Results

**Evaluation metric and datasets.**  We follow [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval) and use the win rate over test dataset to evaluate the quality of the model. We evaluate the win rate over [Anthropic HH](https://huggingface.co/datasets/Anthropic/hh-rlhf), [Slim Orca](https://huggingface.co/datasets/Open-Orca/SlimOrca) and [HEx-PHI](https://huggingface.co/datasets/LLM-Tuning-Safety/HEx-PHI) test prompts.

We give an example on the Llama-3-8b-Instruct model as follows.

<br/><br/>
<div align="center">
    <img alt="seal llama3" src="./docs/seal_llama3.png" style="height: 210px;" />
</div>
<br/><br/>

|   | **Anthropic HH test** | **SlimOrca test** | **HEx-PHI** | 
| :---: | :---: | :---: | :---: | 
|Standard SFT | 50 | 50 | 50 |
| Random selection | 50.78 | 50.8 | 56.31 |
| [DSIR](https://github.com/p-lambda/dsir) | 57.57   | 55.84 | 53.95 |
|[SafeInstr](https://github.com/vinid/safety-tuned-llamas) | 57.97  | 54.22 | 64.49 |
|SEAL | 60.22 | 53.88 | 69.29|
|SEAL+[SafeInstr](https://github.com/vinid/safety-tuned-llamas)| 67.19 | 53.91 | 77.28 |


## Installation

Create conda environment

```bash
conda create -n seal python=3.10
conda activate seal
```

To install the denpendencies, navigate to the root directory and
```bash
pip install -r requirements.txt 
```

> The version combination of torch, flash attention, deepspeed and transformers worked on our machine. You can try other version combination as well.

Then install the SEAL 
```bash
pip install -e .
```

## Running Example

Navigate to scripts folder
```
cd examples/scripts
```

#### Data selector training
In SEAL, we first train a data selector, e.g., with the following script

```bash
deepspeed ../train_sft_selector.py \
    --max_len 2048 \
    --dataset <upper-level safe dataset> \ 
    --new_dataset <original fine-tuning dataset> \ 
    --upperlevel_weight <initial safe loss weight, between (0,1]> \
    --upperlevel_weight_decay <weight decay each epoch> \
    --train_batch_size 64 \
    --micro_train_batch_size 1 \
    --max_samples <dataset size limit>\
    --pretrain <aligned model> \
    --selector_activation <softmax or sigmoid> \
    --selector_name <selector name> \
    --save_steps -1 \
    --logging_steps 1 \
    --eval_steps -1 \
    --zero_stage 3 \
    --max_epochs 3 \
    --bf16 \
    --learning_rate 1e-5 \
    --selector_learning_rate 5e-3 \
    --selector_lr_scheduler <deepspeed lr scheduler, e.g., constant> \
    --lr_scheduler <deepspeed lr scheduler, e.g., cosine> \
    --gradient_checkpointing \
    --flash_attn \
    --lora_rank 16 \
    --lora_alpha 16 \
    --target_modules q_proj v_proj
```

For example, we can run on Llama-3-8b-Instruct with our default setting:
> We provide data selectors trained by us in the ckpt folder. Skip this for a quick run.

```bash
# SEAL data selector training
bash train_selector_llama3.sh
```

#### Fine-tuning stage

Then we run SFT with SEAL data selection
```bash
deepspeed ../train_sft.py \
    --max_len 2048 \
    --dataset <original fine-tuning dataset> \ 
    --selector_path <data selector path>\
    --topp <between (0,1], data selection rate> \
    --train_batch_size 64 \
    --micro_train_batch_size 1 \
    --max_samples  <dataset size limit> \
    --pretrain <initial model>\
    --save_path <save path>\
    --save_steps -1 \
    --logging_steps 1 \
    --eval_steps -1 \
    --zero_stage 3 \
    --max_epochs 3 \
    --bf16 \
    --lr_scheduler <deepspeed lr scheduler, e.g., cosine>\
    --learning_rate 1e-5 \
    --gradient_checkpointing \
    --flash_attn \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules q_proj v_proj
```

For example, the user can run on Llama-3-8b-Instruct with default setting
```bash
# Fine-tuning with data selection
bash train_seal_sft_llama3.sh
```

To train with SFT without data selection, the user just have to set topp as 1. or not specifying the selector_path argument. For example, to run SFT on Llama-3-8b-Instruct without data selection under default setups, use

```bash
# standard SFT on Llama-3-8b-Instruct
bash train_sft_llama3.sh
```
