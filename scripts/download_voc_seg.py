import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms
from torchvision.datasets import VOCSegmentation

voc2012_detection_train = VOCSegmentation(
    root='./data',           # 資料集存放目錄
    year='2012',            # 年份設定為2012
    image_set='train',      # 可選擇 'train', 'trainval', 'val'
    download=True,          # 自動下載
    transform=None,         # 可選的圖像變換
    target_transform=None   # 可選的標籤變換
)

# 選擇前500筆VOC資料
voc_subset = Subset(voc2012_detection_train, list(range(240)))

voc2012_detection_val = VOCSegmentation(
    root='./data',           # 資料集存放目錄
    year='2012',            # 年份設定為2012
    image_set='val',      # 可選擇 'train', 'trainval', 'val'
    download=True,          # 自動下載
    transform=None,         # 可選的圖像變換
    target_transform=None   # 可選的標籤變換
)

voc_subset = Subset(voc2012_detection_val, list(range(60)))