# Deep Learning Assignment 2 Report

## 1. Problem Description

本作業目標為設計一個統一的多任務學習模型，能夠在有限參數（8M 以內）、有限訓練時間（2小時內）、有限推理時間（150ms 以內）下，依序學習三個任務：
- VOC2012 segmentation
- COCO subset detection
- Imagenette classification

並且要求每個任務在經過後續任務訓練後，performance drop 不超過 5%。

---

## 2. Model Architecture

- **Backbone**: MobileNetV3-Small (預訓練, 參數量低)
- **Neck**: 兩層 Conv-BN-ReLU
- **Head**: 統一多任務 head，分別輸出 detection, segmentation, classification
- **參數總數**: 約 3.15M
- **推理時間**: 約 4.2ms (batch=1, 512x512, T4 GPU)

---

## 3. Training Strategy

- **三階段訓練**: 先 segmentation，後 detection (EWC)，最後 classification (replay)
- **Replay Buffer**: detection/classification 階段都 replay segmentation 舊資料
- **EWC**: detection 階段加入 Elastic Weight Consolidation
- **Backbone Freeze**: detection/classification 階段凍結 backbone，僅訓練 head
- **Loss**: detection/classification 階段 loss 加 segmentation replay loss，權重可調
- **訓練時間**: 約 2~3 分鐘（可大幅增加 epochs 以用滿2小時）

---

## 4. Experimental Results

| 任務           | Baseline | After Detection | After Classification | Drop (%) |
|----------------|----------|-----------------|---------------------|----------|
| Segmentation   | 0.159    | 0.035           | 0.040               | 76.6     |
| Detection mAP  | 0.5      | -               | 0.5                 | 0.0      |
| Classification | 1.0      | -               | 1.0                 | 0.0      |

- **參數數量**: 3,152,507
- **推理時間**: 4.2ms
- **訓練時間**: 2.6分鐘

---

## 5. Analysis & Discussion

- **Segmentation drop 仍大於 5%**，即使已經 freeze backbone、加大 replay buffer、提升 replay loss 權重。
- **Detection/Classification 幾乎無 drop**，代表 catastrophic forgetting 主要發生在 segmentation。
- **可能原因**: 
  - 多任務 head 仍有參數干擾
  - replay buffer 仍不足以彌補 feature shift
  - backbone 雖 freeze，但 head 仍被覆蓋
- **改進方向**:
  - 採用更獨立的 multi-head 或 multi-neck
  - 採用 joint training 或 continual multi-task learning
  - 嘗試 LwF、MAS、RWalk 等進階方法

---

## 6. Conclusion

本專案已達成大部分作業要求（參數、推理、訓練時間、detection/classification drop），但 segmentation drop 仍未完全符合 <5% 的標準。已嘗試多種 continual learning 技巧，未來可進一步優化 multi-task 架構與 replay 策略。

---

## 7. 附錄

- 主要程式碼: colab.ipynb, src/models/unified_model.py, src/training/trainer.py
- 詳細 loss curve、訓練 log、完整 config 請見 results/ 資料夾