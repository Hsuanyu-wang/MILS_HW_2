# scripts/eval.py
def evaluate_all_tasks(model_path, data_root):
    """Evaluation script used by TAs"""
    model = torch.load(model_path)
    
    # Load test sets (60 images each)
    det_test = load_coco_test(data_root)
    seg_test = load_voc_test(data_root) 
    cls_test = load_imagenette_test(data_root)
    
    # Compute metrics
    map_score = compute_detection_map(model, det_test)
    miou_score = compute_segmentation_miou(model, seg_test)
    top1_score = compute_classification_top1(model, cls_test)
    
    print(f"mAP: {map_score:.3f}")
    print(f"mIoU: {miou_score:.3f}")  
    print(f"Top-1: {top1_score:.3f}")
    
    return map_score, miou_score, top1_score
