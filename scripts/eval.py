# # scripts/eval.py
# def evaluate_all_tasks(model_path, data_root):
#     """Evaluation script used by TAs"""
#     model = torch.load(model_path)
    
#     # Load test sets (60 images each)
#     det_test = load_coco_test(data_root)
#     seg_test = load_voc_test(data_root) 
#     cls_test = load_imagenette_test(data_root)
    
#     # Compute metrics
#     map_score = compute_detection_map(model, det_test)
#     miou_score = compute_segmentation_miou(model, seg_test)
#     top1_score = compute_classification_top1(model, cls_test)
    
#     print(f"mAP: {map_score:.3f}")
#     print(f"mIoU: {miou_score:.3f}")  
#     print(f"Top-1: {top1_score:.3f}")
    
#     return map_score, miou_score, top1_score

import argparse
import torch
from src.models.unified_model import UnifiedModel
from src.evaluation.metrics import evaluate_all_tasks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--dataroot', required=True) 
    parser.add_argument('--tasks', default='all')
    
    args = parser.parse_args()
    
    # 載入模型
    model = UnifiedModel()
    model.load_state_dict(torch.load(args.weights))
    
    # 評估
    results = evaluate_all_tasks(model, args.dataroot)
    
    print(f"mAP: {results['map']:.4f}")
    print(f"mIoU: {results['miou']:.4f}")
    print(f"Top-1: {results['top1']:.4f}")