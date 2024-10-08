set -x

read -r -d '' training_commands <<EOF
../train_sft.py \
    --seed 42 \
    --max_len 2048 \
    --dataset yahma/alpaca-cleaned \
    --topp 1.\
    --dataset_probs 1. \
    --train_batch_size 64 \
    --micro_train_batch_size 4 \
    --max_samples 51776\
    --pretrain ./ckpt/pyt-dpo \
    --save_path ./ckpt/pyt-ep4-ac\
    --save_steps -1 \
    --logging_steps 1 \
    --eval_steps -1 \
    --zero_stage 3 \
    --max_epochs 4 \
    --bf16 \
    --lr_scheduler constant \
    --learning_rate 1e-5 \
    --gradient_checkpointing \
    --flash_attn 
EOF
    # --selector_path ./ckpt/llama3_softmax_p80_631 \

if [[ ${1} != "slurm" ]]; then
    export PATH=$HOME/.local/bin/:$PATH
    deepspeed $training_commands
fi