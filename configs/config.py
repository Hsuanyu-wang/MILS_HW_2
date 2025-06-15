# configs/config.py
import torch

class Config:
    # Training schedule (根據作業要求的3階段訓練)
    stage1_epochs = 200  # Segmentation only
    stage2_epochs = 15  # Detection with EWC  
    stage3_epochs = 10  # Classification with replay
    
    # Model constraints (作業規定)
    max_params = 8e6  # 8M parameters
    max_inference_time = 150  # ms on T4
    max_training_time = 2 * 3600  # 2 hours
    
    # Training parameters
    lr = 0.001
    batch_size = 16
    num_workers = 2
    
    # Forgetting mitigation
    ewc_lambda = 1000  # EWC regularization strength
    replay_size = 100   # Images per task in replay buffer
    
    # Performance thresholds (作業要求<5%下降)
    max_drop_percent = 5
    
    # Dataset paths
    data_root = './data'
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 如果你想用模組級別的配置，也可以這樣：
stage1_epochs = 200
stage2_epochs = 15
stage3_epochs = 10
ewc_lambda = 1000
replay_size = 100
max_drop_percent = 5
lr = 0.001
batch_size = 16
