# src/datasets/data_loaders.py
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, datasets
from torchvision.datasets import VOCSegmentation
from pycocotools.coco import COCO
from PIL import Image
import os
import numpy as np

def get_transforms():
    """根據作業要求的transforms"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Fixed mask transform for VOC segmentation
    def mask_transform(mask):
        # Resize first as PIL Image
        mask = mask.resize((224, 224), Image.NEAREST)
        # Convert to tensor and handle VOC format
        mask = torch.from_numpy(np.array(mask)).long()
        # Keep 255 as ignore index (don't convert to 0 here)
        return mask
    
    return transform, mask_transform

class COCODetectionDataset(Dataset):
    """基於你的download_coco_det.py的COCO數據集"""
    def __init__(self, root_dir='./data/coco_subset', split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # 載入COCO annotations (根據你的下載檔案)
        ann_file = os.path.join(root_dir, 'annotations', f'instances_{split}2017.json')
        self.coco = COCO(ann_file)
        
        # 你下載的10個類別
        self.categories = ["person", "car", "bicycle", "motorcycle", "airplane",
                          "bus", "train", "truck", "boat", "traffic light"]
        self.cat_ids = self.coco.getCatIds(catNms=self.categories)
        
        # 獲取圖片ID
        img_ids = []
        for cat_id in self.cat_ids:
            img_ids.extend(self.coco.getImgIds(catIds=[cat_id]))
        self.img_ids = list(set(img_ids))
        
        # 根據你的下載數量限制
        if split == 'train':
            self.img_ids = self.img_ids[:240]
        else:  # val
            self.img_ids = self.img_ids[:60]
    
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs([img_id])[0]
        img_path = os.path.join(self.root_dir, f'{self.split}2017', img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        # 取得annotation（這裡僅簡單返回所有該圖的標註，實際可根據需要處理）
        ann_ids = self.coco.getAnnIds(imgIds=[img_id], catIds=self.cat_ids, iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)
        return {
            'image': image,
            'targets': anns,  # 這裡直接返回COCO格式annotation
            'img_id': img_id
        }

class ClassificationDataset(Dataset):
    """包裝ImageFolder，返回dict格式"""
    def __init__(self, image_folder):
        self.image_folder = image_folder
    def __len__(self):
        return len(self.image_folder)
    def __getitem__(self, idx):
        image, label = self.image_folder[idx]
        return {'image': image, 'label': label}

class SegmentationDictWrapper(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
    def __len__(self):
        return len(self.base_dataset)
    def __getitem__(self, idx):
        image, mask = self.base_dataset[idx]
        return {'image': image, 'mask': mask}

def create_datasets_from_downloads():
    """根據你的下載檔案創建數據集"""
    
    transform, mask_transform = get_transforms()
    
    # 1. VOC Segmentation (根據你的download_voc_seg.py)
    voc_train = VOCSegmentation(
        root='./data',
        year='2012',
        image_set='train',
        download=False,
        transform=transform,
        target_transform=mask_transform
    )
    voc_train_subset = Subset(voc_train, list(range(240)))
    voc_train_subset = SegmentationDictWrapper(voc_train_subset)
    
    voc_val = VOCSegmentation(
        root='./data',
        year='2012', 
        image_set='val',
        download=False,
        transform=transform,
        target_transform=mask_transform
    )
    voc_val_subset = Subset(voc_val, list(range(60)))
    voc_val_subset = SegmentationDictWrapper(voc_val_subset)
    
    # 2. Imagenette Classification (根據你的download_imagenette_cls.py)
    imagenette_train = datasets.ImageFolder(
        root='./data/imagenette2-160/train',
        transform=transform
    )
    imagenette_train_subset = Subset(imagenette_train, list(range(240)))
    imagenette_train_dataset = ClassificationDataset(imagenette_train_subset)
    
    imagenette_val = datasets.ImageFolder(
        root='./data/imagenette2-160/val', 
        transform=transform
    )
    imagenette_val_subset = Subset(imagenette_val, list(range(60)))
    imagenette_val_dataset = ClassificationDataset(imagenette_val_subset)
    
    # 3. COCO Detection (根據你的download_coco_det.py)
    coco_train = COCODetectionDataset(
        root_dir='./data/coco_subset',
        split='train',
        transform=transform
    )
    
    coco_val = COCODetectionDataset(
        root_dir='./data/coco_subset', 
        split='val',
        transform=transform
    )
    
    return {
        'seg': {'train': voc_train_subset, 'val': voc_val_subset},
        'cls': {'train': imagenette_train_dataset, 'val': imagenette_val_dataset}, 
        'det': {'train': coco_train, 'val': coco_val}
    }

def detection_collate_fn(batch):
    # Batch is a list of dicts with keys: image, targets, img_id
    images = [item['image'] for item in batch]
    targets = [item['targets'] for item in batch]
    img_ids = [item['img_id'] for item in batch]
    # Stack images, keep targets as list
    images = torch.stack(images, dim=0)
    return {'image': images, 'targets': targets, 'img_id': img_ids}

def create_dataloaders(batch_size=16, num_workers=2):
    """創建DataLoaders"""
    datasets = create_datasets_from_downloads()
    
    dataloaders = {}
    for task in ['seg', 'cls', 'det']:
        if task == 'det':
            # Use custom collate_fn for detection
            dataloaders[task] = {
                'train': DataLoader(
                    datasets[task]['train'],
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    collate_fn=detection_collate_fn
                ),
                'val': DataLoader(
                    datasets[task]['val'], 
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    collate_fn=detection_collate_fn
                )
            }
        else:
            dataloaders[task] = {
                'train': DataLoader(
                    datasets[task]['train'],
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers
                ),
                'val': DataLoader(
                    datasets[task]['val'], 
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers
                )
            }
    
    return dataloaders
