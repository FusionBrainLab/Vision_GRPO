export WANDB_PROJECT="grpo-qwen-2-2B"
export WANDB_ENTITY=""
export WANDB_API_KEY=""
export WANDB_NAME="grpo-qwen-2-2B-v-8gpus-one_reward-synthetic-data-check"  


RUN_NAME="Qwen2-VL-2B-GRPO"
export LOG_PATH="./debug_log_$RUN_NAME.txt"

torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
    run_r1_grpo_vlm.py \
    --deepspeed  configs/zero3.json \
    --config configs/grpo-qwen-2.5-v.yaml \
    --json_data_path "./data/textvqa_cot_train_1_bbox_0.json" \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --save_only_model true
