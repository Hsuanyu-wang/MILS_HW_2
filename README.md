# Unified Multi-Task Learning (DL Assignment 2)

## Overview

本專案為多任務持續學習 (continual learning) 系統，目標為在單一模型下依序學習 segmentation、detection、classification，並盡量減少舊任務的遺忘。

## Features

- MobileNetV3-Small backbone
- Unified multi-task head
- EWC regularization
- Replay buffer (segmentation replay in detection/classification)
- Backbone freeze in later stages
- 自動儲存每次訓練結果於 results/ 子資料夾

## Requirements

- Python 3.8+
- torch, torchvision, numpy, tqdm, matplotlib, pycocotools
- 詳細見 requirements.txt

## Usage

1. 安裝依賴
   ```
   pip install -r requirements.txt
   ```
2. 下載資料集
   ```
   python scripts/download_imagenette_cls.py
   python scripts/download_coco_det.py
   python scripts/download_voc_seg.py
   ```
3. 執行訓練與評估
   - 直接在 colab.ipynb 逐 cell 執行
   - 訓練結果與模型自動儲存於 results/ 子資料夾

## File Structure

- `colab.ipynb`：主訓練與評估 notebook
- `src/`：模型、資料集、訓練、評估原始碼
- `scripts/`：資料集下載、評估腳本
- `results/`：每次訓練的模型與詳細結果
- `configs/`：訓練參數設定
- `DL_Assignment_2.pdf`：作業說明

## Results

- 參數數量：3.15M
- 推理時間：4.2ms
- 訓練時間：2.6分鐘
- Segmentation drop: 76.6%（未達標，已盡力優化）
- Detection/Classification drop: 0%

## Notes

- 若需進一步減少 segmentation drop，建議採用更獨立的 multi-head、joint training 或進階 continual learning 方法。
- 詳細 loss curve 與訓練 log 請見 results/ 子資料夾。