set -x

read -r -d '' training_commands <<EOF
../train_sft_selector.py \
    --seed 42 \
    --max_len 2048 \
    --dataset ./datasets/BlueOrca/train.jsonl \
    --dataset_probs 1. \
    --new_dataset ./datasets/RedOrca/train.jsonl \
    --upperlevel_weight 1. \
    --upperlevel_weight_decay 0.02 \
    --train_batch_size 64 \
    --micro_train_batch_size 4 \
    --max_samples 112000 \
    --pretrain ./ckpt/pyt-ep4-ac \
    --selector_name pyt \
    --ref_constant 0. \
    --selector_activation softmax \
    --save_steps -1 \
    --logging_steps 1 \
    --eval_steps -1 \
    --zero_stage 3 \
    --max_epochs 2 \
    --bf16 \
    --learning_rate 1e-5 \
    --selector_learning_rate 4e-3 \
    --selector_lr_scheduler constant \
    --lr_scheduler constant \
    --gradient_checkpointing \
    --flash_attn 
EOF
    # --flash_attn 
    # -dataset Dahoas/full-hh-rlhf,./datasets/SlimOrcaEn/train.jsonl \ ./datasets/RedOrca/train.jsonl
    # microsoft/Phi-3-mini-128k-instruct
    #     --lora_rank 16 \
     #   --lora_alpha 16 \
     #   --target_modules qkv_proj o_proj


if [[ ${1} != "slurm" ]]; then
    export PATH=$HOME/.local/bin/:$PATH
    deepspeed $training_commands
fi
