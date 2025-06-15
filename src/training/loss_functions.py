import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Focal Loss for detection task"""
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

class UncertaintyWeightedLoss(nn.Module):
    """不確定性加權多任務損失"""
    def __init__(self):
        super().__init__()
        # 可學習的任務不確定性參數
        self.log_vars = nn.Parameter(torch.zeros(3))
        
    def forward(self, det_loss, seg_loss, cls_loss):
        # 使用不確定性加權
        precision_det = torch.exp(-self.log_vars[0])
        precision_seg = torch.exp(-self.log_vars[1])
        precision_cls = torch.exp(-self.log_vars[2])
        
        loss = (precision_det * det_loss + self.log_vars[0] +
                precision_seg * seg_loss + self.log_vars[1] +
                precision_cls * cls_loss + self.log_vars[2])
        
        return loss

class MultiTaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.focal_loss = FocalLoss()
        self.seg_loss = nn.CrossEntropyLoss()
        self.cls_loss = nn.CrossEntropyLoss()
        self.uncertainty_loss = UncertaintyWeightedLoss()
        
    def forward(self, predictions, targets):
        det_loss = F.mse_loss(predictions['detection'], targets['detection'])
        seg_loss = self.seg_loss(predictions['segmentation'], targets['segmentation'])
        cls_loss = self.cls_loss(predictions['classification'], targets['classification'])
        
        total_loss = self.uncertainty_loss(det_loss, seg_loss, cls_loss)
        
        return {
            'total_loss': total_loss,
            'det_loss': det_loss,
            'seg_loss': seg_loss,
            'cls_loss': cls_loss
        }
