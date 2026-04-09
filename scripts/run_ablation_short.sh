#!/bin/bash
# run_ablation_short.sh

BASE_OUT_DIR="./output/ablation_study260321_1856"
mkdir -p $BASE_OUT_DIR

# 实验列表
experiments=(
    "full_model"        ""
    #"no_revin"          "--no_revin"
    #"no_cnn"            "--no_cnn"
    #"no_attention"      "--no_attention"
    #"no_trend"          "--no_trend"
    #"standard_loss"     "--no_weighted_loss"
    #"only_trend"        "--no_cnn --no_attention"  # <--- [新增] 只有trend的消融实验
)

for ((i=0; i<${#experiments[@]}; i+=2)); do
    CASE_NAME=${experiments[i]}
    EXTRA_FLAG=${experiments[i+1]}
    
    CURR_OUT_DIR="$BASE_OUT_DIR/$CASE_NAME"
    mkdir -p $CURR_OUT_DIR
    
    echo ">>> Starting Ablation: $CASE_NAME using model: tqnet_enhanced"

    # 注意：这里的 $EXTRA_FLAG 没有加双引号，这样即使里面包含多个参数（如 --no_cnn --no_attention），
    # bash 也会正确地把它们拆分成独立的参数传给 python
    python main.py \
        --mode short \
        --model tqnet_enhanced \
        --out_dir $CURR_OUT_DIR \
        --epochs 400 \
        --patience 30 \
        --lr 0.0001 \
        --batch_size 32 \
        $EXTRA_FLAG \
        2>&1 | tee $CURR_OUT_DIR/train_log.txt
done