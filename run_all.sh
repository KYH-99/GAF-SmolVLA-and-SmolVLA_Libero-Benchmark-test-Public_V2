#!/bin/bash
# Master execution script for SmolVLA x GAF x GAF-Success experiment
# Runs Phase 1 through 6 sequentially.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gaf-exp
cd ~/experiments/gaf-comparison

mkdir -p logs

echo "================================================"
echo "  STARTING FULL EXPERIMENT PIPELINE"
echo "  Date: $(date)"
echo "================================================"

echo "[1/6] Running Phase 1: Baseline..."
python 01_baseline.py > logs/phase1_baseline.log 2>&1
echo "  Phase 1 done. Date: $(date)"

echo "[2/6] Running Phase 2: Rollout Collection..."
python 02_collect.py > logs/phase2_collect.log 2>&1
echo "  Phase 2 done. Date: $(date)"

echo "[3/6] Running Phase 3: Critic Training..."
python 03_train_critics.py > logs/phase3_train.log 2>&1
echo "  Phase 3 done. Date: $(date)"

echo "[4/6] Running Phase 4 & 5: GAF Evaluation..."
python 04_eval_gaf.py > logs/phase4_5_eval.log 2>&1
echo "  Phase 4/5 done. Date: $(date)"

echo "[6/6] Running Phase 6: Analysis & PPTX Generation..."
python 06_analyze_and_slides.py > logs/phase6_analyze.log 2>&1
echo "  Phase 6 done. Date: $(date)"

echo "================================================"
echo "  EXPERIMENT COMPLETE!"
echo "  Date: $(date)"
echo "================================================"
