# MILS_HW2: Unified Multi-Task Learning

## Overview
This project implements a unified deep learning model for three tasks: semantic segmentation (VOC), object detection (COCO subset), and image classification (Imagenette). The model is trained in three stages with forgetting mitigation (EWC, replay buffer) and is constrained to <8M parameters and <150ms inference time.

## Directory Structure
- `colab.ipynb`: Main notebook for training and evaluation
- `src/`: Source code (models, datasets, training, evaluation)
- `scripts/`: Dataset download and evaluation scripts
- `configs/`: Configuration
- `data/`: Downloaded datasets (after running scripts)
- `checkpoints/`: Saved models

## Setup
1. Clone the repo and enter the `MILS_HW2` directory.
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Download datasets:
   ```bash
   python scripts/download_imagenette_cls.py
   python scripts/download_coco_det.py
   python scripts/download_voc_seg.py
   ```

## Running the Notebook
Open `colab.ipynb` in Jupyter or Colab and run all cells. The notebook will:
- Verify datasets
- Initialize model and dataloaders
- Train in three stages (segmentation, detection with EWC, classification with replay)
- Evaluate and save results

## Evaluation
To evaluate a trained model:
```bash
python scripts/eval.py --weights checkpoints/final_model.pt --dataroot data --tasks all
```

## Requirements & Constraints
- Model parameters: <8M
- Inference time: <150ms (T4 GPU)
- Training time: <2 hours
- Performance drop after all stages: <5% on all tasks

## Contact
For questions, contact: [Your Name/Email]
