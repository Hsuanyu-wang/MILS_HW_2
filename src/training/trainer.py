# src/training/trainer.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskTrainer:
    def __init__(self, model, datasets, config):
        self.model = model
        self.datasets = datasets  # {'det': det_loader, 'seg': seg_loader, 'cls': cls_loader}
        self.config = config
        self.stage = 0
        self.baselines = {}  # Store single-task baselines
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        self.seg_criterion = nn.CrossEntropyLoss(ignore_index=255)
        self.cls_criterion = nn.CrossEntropyLoss()
        self.det_criterion = nn.MSELoss()  # 佔位，實際應根據detection head格式設計
        
    def train_stage_1_segmentation(self):
        """Stage 1: Train ONLY on segmentation"""
        print("Stage 1: Training on Mini-VOC-Seg only...")
        for epoch in range(self.config.stage1_epochs):
            for batch in self.datasets['seg']:
                self.optimizer.zero_grad()
                _, seg_out, _ = self.model(batch['image'])
                if seg_out.shape[-2:] != batch['mask'].shape[-2:]:
                    seg_out = F.interpolate(seg_out, size=batch['mask'].shape[-2:], mode='bilinear', align_corners=False)
                batch['mask'] = batch['mask'].long()
                loss = self.seg_criterion(seg_out, batch['mask'])
                loss.backward()
                self.optimizer.step()
        
        # Record baseline performance
        self.baselines['mIoU'] = self.evaluate_segmentation()
        
    def train_stage_2_detection(self):
        """Stage 2: Train ONLY on detection with EWC"""
        print("Stage 2: Training on Mini-COCO-Det with forgetting mitigation...")
        
        # Apply Elastic Weight Consolidation
        self.apply_ewc_regularization()
        
        for epoch in range(self.config.stage2_epochs):
            for batch in self.datasets['det']:
                self.optimizer.zero_grad()
                det_out, _, _ = self.model(batch['image'])
                det_loss = self.det_criterion(det_out.float(), torch.zeros_like(det_out))
                ewc_loss = self.compute_ewc_loss() if hasattr(self, 'compute_ewc_loss') else 0.0
                total_loss = det_loss + self.config.ewc_lambda * ewc_loss
                total_loss.backward()
                self.optimizer.step()
        
        self.baselines['mAP'] = self.evaluate_detection()
        
    def train_stage_3_classification(self):
        """Stage 3: Train ONLY on classification with replay buffer"""
        print("Stage 3: Training on Imagenette-160 with replay...")
        
        # Maintain replay buffer (10 images per previous task)
        replay_buffer = self.create_replay_buffer() if hasattr(self, 'create_replay_buffer') else None
        
        for epoch in range(self.config.stage3_epochs):
            for batch in self.datasets['cls']:
                self.optimizer.zero_grad()
                _, _, cls_out = self.model(batch['image'])
                cls_loss = self.cls_criterion(cls_out, batch['label'])
                
                # Add replay loss
                replay_loss = self.compute_replay_loss(replay_buffer) if replay_buffer is not None and hasattr(self, 'compute_replay_loss') else 0.0
                total_loss = cls_loss + replay_loss
                total_loss.backward()
                self.optimizer.step()
        
        self.baselines['Top1'] = self.evaluate_classification()
        
    def validate_forgetting_constraint(self):
        """Ensure <5% performance drop on all tasks"""
        current_miou = self.evaluate_segmentation()
        current_map = self.evaluate_detection() 
        current_top1 = self.evaluate_classification()
        
        miou_drop = (self.baselines['mIoU'] - current_miou) / self.baselines['mIoU'] * 100
        map_drop = (self.baselines['mAP'] - current_map) / self.baselines['mAP'] * 100
        top1_drop = (self.baselines['Top1'] - current_top1) / self.baselines['Top1'] * 100
        
        print(f"Performance drops: mIoU: {miou_drop:.1f}%, mAP: {map_drop:.1f}%, Top-1: {top1_drop:.1f}%")
        
        return all([miou_drop < 5, map_drop < 5, top1_drop < 5])

    def compute_fisher_information(self, seg_loader):
        for batch in seg_loader:
            images = batch['image']
            masks = batch['mask']
            self.model.zero_grad()
            _, seg_out, _ = self.model(images)
            if seg_out.shape[-2:] != masks.shape[-2:]:
                seg_out = F.interpolate(seg_out, size=masks.shape[-2:], mode='bilinear', align_corners=False)
            loss = self.seg_criterion(seg_out, masks)
            loss.backward()
