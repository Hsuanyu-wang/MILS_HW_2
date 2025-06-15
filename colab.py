# notebooks/colab.ipynb
import os
# Change to the MILS_HW2 directory first
os.chdir('../MILS_HW2')

# Cell 2: Verify downloads
# Add a verification step
print("Verifying downloads...")
import os
data_paths = {
    'seg': './data/VOCdevkit/VOC2012',
    'det': './data/coco_subset',
    'cls': './data/imagenette2-160'
}

for task, path in data_paths.items():
    if os.path.exists(path):
        print(f"{task} dataset found at {path}")
    else:
        print(f"WARNING: {task} dataset not found at {path}")
        
# Cell 3: Model and Data initialization 
from src.models.unified_model import UnifiedModel
from src.datasets.data_loaders import create_dataloaders
from configs.config import Config  # 使用Config類

# 初始化配置
config = Config()  # 創建Config實例，不是模組

# 初始化模型
model = UnifiedModel()
print(f"Total parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

# 創建數據載入器 (根據你的下載檔案)
print("Loading datasets...")
dataloaders = create_dataloaders(
    batch_size=config.batch_size,
    num_workers=config.num_workers
)
print("dataloaders", dataloaders)

# 準備datasets字典給trainer使用 (只用train set)
datasets = {
    'seg': dataloaders['seg']['train'],
    'det': dataloaders['det']['train'], 
    'cls': dataloaders['cls']['train']
}
print("datasets:\n", datasets)
print("Datasets loaded successfully!")
print(f"Detection batches: {len(datasets['det'])}")
print(f"Segmentation batches: {len(datasets['seg'])}")
print(f"Classification batches: {len(datasets['cls'])}")

# Cell 4: Three-stage training
from src.training.trainer import MultiTaskTrainer
trainer = MultiTaskTrainer(model, datasets, config)

# Stage 1: Segmentation baseline
trainer.train_stage_1_segmentation()

# Stage 2: Detection with EWC
trainer.train_stage_2_detection()  

# Stage 3: Classification with replay
trainer.train_stage_3_classification()

# Validate forgetting constraint
success = trainer.validate_forgetting_constraint()
print(f"Forgetting constraint satisfied: {success}")

