#!/bin/bash

# 确保遇到错误就停止
set -e

# 进入项目根目录 (兼容你提供的写法)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 环境变量与全局配置
export OMP_NUM_THREADS=4
DATA_PATH="./data/all_energy_clean_modified.csv"

# 生成本次运行的统一时间戳 (精确到秒防止冲突)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ROOT_OUT_DIR="./output_results_$TIMESTAMP"

echo "=========================================================="
echo ">>> 启动长短时预测 Pipeline <<<"
echo ">>> 本次运行根目录: $ROOT_OUT_DIR"
echo "=========================================================="

# 创建根目录
mkdir -p "$ROOT_OUT_DIR"

# ==========================================================
# 实验 1: 短期预测 (Short-Term: E-TQNet)
# ==========================================================
# TASK1_DIR="$ROOT_OUT_DIR/Task1_ShortTerm"
# mkdir -p "$TASK1_DIR"
# echo -e "\n--- [Task 1] 开始短期预测 (168h -> 24h) ---"
# echo ">> 输出与日志将保存在: $TASK1_DIR"

# python main.py \
#     --mode short \
#     --model tqnet_enhanced \
#     --file_path "$DATA_PATH" \
#     --out_dir "$TASK1_DIR" \
#     --seq_len 168 \
#     --pred_len 24 \
#     --d_model 64 \
#     --epochs 100 \
#     --batch_size 32 \
#     --lr 0.001 \
#     2>&1 | tee "$TASK1_DIR/train.log"

# ==========================================================
# 实验 2: 长期预测 (Long-Term: FA-DualTQNet)
# ==========================================================
TASK2_DIR="$ROOT_OUT_DIR/Task2_LongTerm"
mkdir -p "$TASK2_DIR"
echo -e "\n--- [Task 2] 开始长期预测 (336h -> 720h) ---"
echo ">> 输出与日志将保存在: $TASK2_DIR"

python main.py \
    --mode long \
    --model tqnet_dual \
    --file_path "$DATA_PATH" \
    --out_dir "$TASK2_DIR" \
    --seq_len 336 \
    --pred_len 720 \
    --d_model 128 \
    --epochs 400 \
    --batch_size 64 \
    --lr 0.0001 \
    --patience 30 \
    2>&1 | tee "$TASK2_DIR/train.log"

echo -e "\n=========================================================="
echo "   All tasks completed successfully!"
echo "   所有结果、图片和独立日志已归档至: $ROOT_OUT_DIR"
echo "=========================================================="