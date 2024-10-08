set -x

read -r -d '' training_commands <<EOF
../train_sft_selector.py \
    --seed 42 \
    --max_len 2048 \
    --dataset ./datasets/BlueOrca/train.jsonl \
    --dataset_probs 1. \
    --new_dataset ./datasets/RedOrca/train.jsonl \
    --upperlevel_weight 1. \
    --upperlevel_weight_decay 0.03 \
    --train_batch_size 64 \
    --micro_train_batch_size 1 \
    --max_samples 112000 \
    --pretrain meta-llama/Meta-Llama-3-8B-Instruct \
    --ref_constant 0. \
    --selector_activation softmax \
    --selector_name llama3 \
    --save_steps -1 \
    --logging_steps 1 \
    --eval_steps -1 \
    --zero_stage 3 \
    --max_epochs 3 \
    --bf16 \
    --learning_rate 1e-5 \
    --selector_learning_rate 5e-3 \
    --selector_lr_scheduler constant \
    --lr_scheduler constant \
    --gradient_checkpointing \
    --flash_attn \
    --lora_rank 16 \
    --lora_alpha 16 \
    --target_modules q_proj v_proj
EOF
    # --flash_attn 
    # -dataset Dahoas/full-hh-rlhf,./datasets/SlimOrcaEn/train.jsonl \ ./datasets/RedOrca/train.jsonl
    # microsoft/Phi-3-mini-128k-instruct
    #     --lora_rank 16 \
     #   --lora_alpha 16 \
     #   --target_modules qkv_proj o_proj


if [[ ${1} != "slurm" ]]; then
    export PATH=$HOME/.local/bin/:$PATH
    deepspeed --num_gpus 4 $training_commands
fi