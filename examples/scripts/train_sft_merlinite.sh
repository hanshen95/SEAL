set -x

read -r -d '' training_commands <<EOF
../train_sft.py \
    --seed 42 \
    --max_len 2048 \
    --dataset ./datasets/RedOrca/train.jsonl \
    --topp 1. \
    --dataset_probs 1. \
    --train_batch_size 64 \
    --micro_train_batch_size 4 \
    --max_samples 112000\
    --pretrain ibm/merlinite-7b \
    --save_path ./ckpt/merlin-random80-ep3-ro \
    --save_steps -1 \
    --logging_steps 1 \
    --eval_steps -1 \
    --zero_stage 3 \
    --max_epochs 3 \
    --bf16 \
    --learning_rate 1e-5 \
    --gradient_checkpointing \
    --target_module q_proj v_proj \
    --lora_rank 16 \
    --lora_alpha 16
EOF

if [[ ${1} != "slurm" ]]; then
    export PATH=$HOME/.local/bin/:$PATH
    deepspeed $training_commands
fi
