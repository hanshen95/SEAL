set -x

read -r -d '' training_commands <<EOF
../train_sft.py \
    --seed 42 \
    --max_len 2048 \
    --dataset ./datasets/RedOrca/train.jsonl \
    --selector_path ./ckpt/seal_selector_llama3.pt \
    --topp 0.8 \
    --dataset_probs 1. \
    --train_batch_size 64 \
    --micro_train_batch_size 1 \
    --max_samples 112000\
    --pretrain meta-llama/Meta-Llama-3-8B-Instruct \
    --save_path ./ckpt/llama3-llama380-ep2-ro\
    --save_steps -1 \
    --logging_steps 1 \
    --eval_steps -1 \
    --zero_stage 3 \
    --max_epochs 2 \
    --bf16 \
    --lr_scheduler constant \
    --learning_rate 1e-5 \
    --gradient_checkpointing \
    --flash_attn \
    --lora_rank 16 \
    --lora_alpha 16 \
    --target_modules q_proj v_proj 
EOF

if [[ ${1} != "slurm" ]]; then
    export PATH=$HOME/.local/bin/:$PATH
    deepspeed $training_commands
fi
