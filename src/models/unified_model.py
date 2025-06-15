# src/models/unified_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small

class UnifiedMultiTaskHead(nn.Module):
    """統一的多任務頭部，只有2-3層設計"""
    def __init__(self, in_channels=256, num_classes_det=10, num_classes_seg=21, num_classes_cls=10):
        super().__init__()
        
        # 第1層：共享特徵處理層
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # 第2層：任務特定輸出層
        # Detection head: 輸出 [cx, cy, w, h, conf, class] * num_classes
        self.detection_head = nn.Conv2d(128, num_classes_det * 6, kernel_size=1)
        
        # Segmentation head: 輸出每個像素的類別
        self.segmentation_head = nn.Conv2d(128, num_classes_seg, kernel_size=1)
        
        # Classification head: 全局平均池化 + 全連接層
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes_cls)
        )
        
        # 第3層：上採樣層（僅用於分割任務）
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
        
    def forward(self, x):
        # 第1層：共享特徵處理
        shared_features = self.shared_conv(x)  # [N, 128, H/16, W/16]
        
        # 第2層：任務特定輸出
        # Detection output: [N, classes*6, H/16, W/16]
        det_out = self.detection_head(shared_features)
        
        # Segmentation output: [N, classes, H/16, W/16] -> [N, classes, H, W]
        seg_out = self.segmentation_head(shared_features)
        seg_out = self.upsample(seg_out)  # 第3層：上採樣到原始尺寸
        
        # Classification output: [N, classes]
        cls_out = self.classification_head(shared_features)
        
        return det_out, seg_out, cls_out

class UnifiedModel(nn.Module):
    """統一的多任務學習模型"""
    def __init__(self):
        super().__init__()
        
        # Backbone: MobileNetV3-Small (預訓練，約2.5M參數)
        self.backbone = mobilenet_v3_small(pretrained=True)
        # 移除原始分類器，保留特徵提取部分
        self.backbone.classifier = nn.Identity()
        
        # 獲取backbone輸出通道數
        backbone_out_channels = 576  # MobileNetV3-Small的輸出通道數
        
        # Neck: 特徵融合層（2層ConvBNReLU）
        self.neck = nn.Sequential(
            nn.Conv2d(backbone_out_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Head: 統一的多任務頭部（2-3層）
        self.head = UnifiedMultiTaskHead(
            in_channels=256,
            num_classes_det=10,  # COCO subset的10個類別
            num_classes_seg=21,  # VOC的21個類別（包括背景）
            num_classes_cls=10   # Imagenette的10個類別
        )
        
        # 初始化權重
        self._initialize_weights()
        
    def _initialize_weights(self):
        """初始化新增層的權重"""
        for m in [self.neck, self.head]:
            for module in m.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.BatchNorm2d):
                    nn.init.constant_(module.weight, 1)
                    nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, 0, 0.01)
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """前向傳播"""
        # Backbone特徵提取
        features = self.backbone.features(x)  # [N, 576, H/16, W/16]
        
        # Neck特徵融合
        neck_features = self.neck(features)   # [N, 256, H/16, W/16]
        
        # Head多任務輸出
        det_out, seg_out, cls_out = self.head(neck_features)
        
        return det_out, seg_out, cls_out
    
    def get_parameter_count(self):
        """獲取模型參數數量"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'total_M': total_params / 1e6,
            'trainable_M': trainable_params / 1e6
        }
    
    def freeze_backbone(self):
        """凍結backbone參數"""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """解凍backbone參數"""
        for param in self.backbone.parameters():
            param.requires_grad = True

# 測試函數
def test_model():
    """測試模型的基本功能"""
    model = UnifiedModel()
    
    # 測試輸入
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    
    # 前向傳播
    with torch.no_grad():
        det_out, seg_out, cls_out = model(input_tensor)
    
    print("Model Architecture Test:")
    print(f"Input shape: {input_tensor.shape}")
    print(f"Detection output shape: {det_out.shape}")      # [N, 60, H/16, W/16]
    print(f"Segmentation output shape: {seg_out.shape}")   # [N, 21, H, W]
    print(f"Classification output shape: {cls_out.shape}") # [N, 10]
    
    # 參數統計
    param_info = model.get_parameter_count()
    print(f"\nParameter Count:")
    print(f"Total parameters: {param_info['total_M']:.2f}M")
    print(f"Trainable parameters: {param_info['trainable_M']:.2f}M")
    
    return model

if __name__ == "__main__":
    test_model()
