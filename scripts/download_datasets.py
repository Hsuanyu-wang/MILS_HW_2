# scripts/download_datasets.py
import os
import sys
sys.path.append('..')

def download_all_datasets():
    """Download all three datasets for the multi-task challenge"""
    print("Starting dataset downloads...")
    
    # Create main data directory
    os.makedirs('./data', exist_ok=True)
    
    # Download Mini-COCO-Det (240 train, 60 val)
    print("1/3 Downloading Mini-COCO-Det...")
    exec(open('download_coco_det.py').read())
    
    # Download Mini-VOC-Seg (240 train, 60 val)  
    print("2/3 Downloading Mini-VOC-Seg...")
    exec(open('download_voc_seg.py').read())
    
    # Download Imagenette-160 (240 train, 60 val)
    print("3/3 Downloading Imagenette-160...")
    exec(open('download_imagenette_cls.py').read())
    
    print("All datasets downloaded successfully!")
    print("Total size: ~120MB as specified")

if __name__ == "__main__":
    download_all_datasets()
