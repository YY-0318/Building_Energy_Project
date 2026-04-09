#!/bin/bash

# ==========================================================
# 1. 自动定位到项目根目录
# ==========================================================
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export OMP_NUM_THREADS=4
# 定义统一的输出根目录，按日期归档，所有8个模型的结果都会在这个文件夹里
OUT_ROOT="./output/all_baselines_$(date +%y%m%d)"
mkdir -p "$OUT_ROOT"

echo "======================================================="
echo ">>> 启动全矩阵 Baseline 对比实验流水线 (共8个模型) <<<"
echo "======================================================="

# ==========================================================
# 第一部分: TSLib SOTA 模型 (具有相同的超参数设置，使用循环)
# ==========================================================
echo "--- 阶段 1/2: 运行 TSLib Transformer 系列模型 ---"
TSLIB_MODELS=("iTransformer" "Autoformer" "Crossformer" "FEDformer")

for m in "${TSLIB_MODELS[@]}"; do
    MODEL_OUT_DIR="$OUT_ROOT/$m"
    mkdir -p "$MODEL_OUT_DIR"
    echo ">> [TSLib] 正在运行: $m..."
    
    python main.py --mode short --model $m \
        --out_dir "$MODEL_OUT_DIR" \
        --patience 30 --batch_size 32 --epochs 300 --lr 0.0001 \
        2>&1 | tee "$MODEL_OUT_DIR/train.log"
done

# ==========================================================
# 第二部分: 经典模型与其他 Baseline (具有各自定制的超参数)
# ==========================================================
echo "--- 阶段 2/2: 运行经典与专项 Baseline 模型 ---"

# 1. DLinear
DLINEAR_DIR="$OUT_ROOT/dlinear"
mkdir -p "$DLINEAR_DIR"
echo ">> [Classic] 正在运行: DLinear..."
python main.py --mode short --model dlinear --out_dir "$DLINEAR_DIR" \
    --epochs 300 --lr 0.0001 --batch_size 30 --weight_decay 0.0001 \
    2>&1 | tee "$DLINEAR_DIR/train.log"

# 2. LSTM (注意这里 lr=0.001)
LSTM_DIR="$OUT_ROOT/lstm"
mkdir -p "$LSTM_DIR"
echo ">> [Classic] 正在运行: LSTM..."
python main.py --mode short --model lstm --out_dir "$LSTM_DIR" \
    --epochs 300 --lr 0.0001 --batch_size 30 --weight_decay 0.0001 \
    2>&1 | tee "$LSTM_DIR/train.log"

# 3. PatchTST (SOTA 对比)
PATCHTST_DIR="$OUT_ROOT/patchtst"
mkdir -p "$PATCHTST_DIR"
echo ">> [Classic] 正在运行: PatchTST..."
python main.py --mode short --model patchtst --out_dir "$PATCHTST_DIR" \
    --epochs 300 --lr 0.0001 --batch_size 30 \
    2>&1 | tee "$PATCHTST_DIR/train.log"

# 4. TQNet Vanilla (你原本模型的基础退化版本，用于做消融对比)
TQNET_DIR="$OUT_ROOT/tqnet_vanilla"
mkdir -p "$TQNET_DIR"
echo ">> [Classic] 正在运行: TQNet Vanilla..."
python main.py --mode short --model tqnet_vanilla --out_dir "$TQNET_DIR" \
    --epochs 300 --lr 0.0001 --batch_size 30 --weight_decay 0.0001 \
    2>&1 | tee "$TQNET_DIR/train.log"

echo "======================================================="
echo ">>> 🎉 所有 8 个对比实验运行完毕！"
echo ">>> 结果已全部分类保存在: $OUT_ROOT"
echo "======================================================="