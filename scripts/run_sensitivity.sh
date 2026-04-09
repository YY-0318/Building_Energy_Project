#!/bin/bash
# 进入项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 环境变量与全局配置
export OMP_NUM_THREADS=4
DATA_PATH="./data/all_energy_clean_modified.csv"
TIMESTAMP=$(date +%Y%m%d_%H%M)
ROOT_OUT_DIR="./output/sensitivity_analysis_$TIMESTAMP"

# ==========================================
# 统一超参数设置 (集中管理，方便修改)
# ==========================================
MODEL_NAME="tqnet_enhanced"
EPOCHS=400
PATIENCE=30

mkdir -p "$ROOT_OUT_DIR"
echo "=========================================================="
echo ">>> 启动全参数敏感性分析 (模型: $MODEL_NAME) <<<"
echo ">>> 日志根目录: $ROOT_OUT_DIR"
echo "=========================================================="

# ==========================================================
# 实验 1: 历史观测窗口 seq_len (分析长程依赖能力)
# ==========================================================
echo "--- 开始分析 1/6: 历史观测窗口 (seq_len) ---"
for sl in 24 168; do
    OUT_PATH="$ROOT_OUT_DIR/Sens_SeqLen_$sl"
    mkdir -p "$OUT_PATH"
    echo ">> 正在运行 seq_len = $sl"
    
    python main.py --mode short --model "$MODEL_NAME" \
        --seq_len "$sl" --file_path "$DATA_PATH" --out_dir "$OUT_PATH" \
        --epochs "$EPOCHS" --patience "$PATIENCE" \
        2>&1 | tee "$OUT_PATH/train.log"
done

# # ==========================================================
# # 实验 2: 注意力头数 n_heads (分析多特征解耦能力)
# # ==========================================================
# echo "--- 开始分析 2/6: 注意力头数 (n_heads) ---"
# for nh in 2 4 8 ; do
#     OUT_PATH="$ROOT_OUT_DIR/Sens_Heads_$nh"
#     mkdir -p "$OUT_PATH"
#     echo ">> 正在运行 n_heads = $nh"
    
#     python main.py --mode short --model "$MODEL_NAME" \
#         --n_heads "$nh" --file_path "$DATA_PATH" --out_dir "$OUT_PATH" \
#         --epochs "$EPOCHS" --patience "$PATIENCE" \
#         2>&1 | tee "$OUT_PATH/train.log"
# done

# # ==========================================================
# # 实验 3: CNN 卷积核大小 kernel_size (分析局部突变捕捉能力)
# # 注意：卷积核大小必须是奇数，以便对称 padding
# # ==========================================================
# echo "--- 开始分析 3/6: CNN 卷积核大小 (kernel_size) ---"
# for ks in 3 5 7 ; do
#     OUT_PATH="$ROOT_OUT_DIR/Sens_Kernel_$ks"
#     mkdir -p "$OUT_PATH"
#     echo ">> 正在运行 kernel_size = $ks"
    
#     python main.py --mode short --model "$MODEL_NAME" \
#         --kernel_size "$ks" --file_path "$DATA_PATH" --out_dir "$OUT_PATH" \
#         --epochs "$EPOCHS" --patience "$PATIENCE" \
#         2>&1 | tee "$OUT_PATH/train.log"
# done

# ==========================================================
# 实验 4: 模型维度 d_model 分析 (特征表达容量)
# ==========================================================
# echo "--- 开始分析 4/6: 模型维度 (d_model) ---"
# for d in 16 32 64 128; do
#     OUT_PATH="$ROOT_OUT_DIR/Sens_Dim_$d"
#     mkdir -p "$OUT_PATH"
#     echo ">> 正在运行 d_model = $d"

#     python main.py --mode short --model "$MODEL_NAME" \
#         --d_model "$d" --file_path "$DATA_PATH" --out_dir "$OUT_PATH" \
#         --epochs "$EPOCHS" --patience "$PATIENCE" \
#         2>&1 | tee "$OUT_PATH/train.log"
# done

# ==========================================================
# 实验 5: 温度系数 tau 分析 (注意力平滑度控制)
# ==========================================================
# echo "--- 开始分析 5/6: 温度系数 (tau) ---"
# for t in 0.25; do
#     OUT_PATH="$ROOT_OUT_DIR/Sens_Tau_$t"
#     mkdir -p "$OUT_PATH"
#     echo ">> 正在运行 tau = $t"

#     python main.py --mode short --model "$MODEL_NAME" \
#         --tau "$t" --file_path "$DATA_PATH" --out_dir "$OUT_PATH" \
#         --epochs "$EPOCHS" --patience "$PATIENCE" \
#         2>&1 | tee "$OUT_PATH/train.log"
# done

# # ==========================================================
# # 实验 6: 损失权重 loss_weight 分析 (关键时段关注度)
# # ==========================================================
# echo "--- 开始分析 6/6: 损失权重 (loss_weight) ---"
# for lw in 1.0 5.0 10.0; do
#     OUT_PATH="$ROOT_OUT_DIR/Sens_Weight_$lw"
#     mkdir -p "$OUT_PATH"
#     echo ">> 正在运行 loss_weight = $lw"

#     python main.py --mode short --model "$MODEL_NAME" \
#         --loss_weight "$lw" --file_path "$DATA_PATH" --out_dir "$OUT_PATH" \
#         --epochs "$EPOCHS" --patience "$PATIENCE" \
#         2>&1 | tee "$OUT_PATH/train.log"
# done

echo "=========================================================="
echo ">>> 所有敏感性分析实验全部完成！ <<<"
echo ">>> 日志和模型均保存在: $ROOT_OUT_DIR"
echo "=========================================================="