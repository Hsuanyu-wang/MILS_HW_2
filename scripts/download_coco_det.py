from pycocotools.coco import COCO
import requests
import os
import shutil
import zipfile

def download_coco_subset(split="train", num_samples=240):
    # Create directories
    base_dir = "./data/coco_subset"
    images_dir = os.path.join(base_dir, f"{split}2017")
    annot_dir = os.path.join(base_dir, "annotations")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(annot_dir, exist_ok=True)
    
    # Download and extract annotations if not already present
    annotation_zip = os.path.join(base_dir, "annotations.zip")
    if not os.path.exists(os.path.join(annot_dir, "instances_train2017.json")):
        # Download annotations
        print("Downloading COCO annotations...")
        annotation_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        response = requests.get(annotation_url, stream=True)
        with open(annotation_zip, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        # Extract annotations
        print("Extracting annotations...")
        with zipfile.ZipFile(annotation_zip, 'r') as zip_ref:
            zip_ref.extractall(base_dir)
        
        # Clean up zip file
        os.remove(annotation_zip)
    
    # Classes we want to download
    categories = ["person", "car", "bicycle", "motorcycle", "airplane", 
                 "bus", "train", "truck", "boat", "traffic light"]
    
    # Initialize COCO API with local annotation file
    annotation_file = os.path.join(annot_dir, f"instances_{split}2017.json")
    coco = COCO(annotation_file)
    
    # Get category IDs
    cat_ids = coco.getCatIds(catNms=categories)
    
    # Get image IDs for these categories
    img_ids = []
    for cat_id in cat_ids:
        img_ids.extend(coco.getImgIds(catIds=[cat_id]))
    img_ids = list(set(img_ids))[:num_samples]  # Remove duplicates and limit samples
    
    # Download images
    print(f"Downloading {len(img_ids)} images for {split} set...")
    for i, img_id in enumerate(img_ids):
        img = coco.loadImgs([img_id])[0]
        img_url = img['coco_url']
        file_name = img['file_name']
        
        # Download image if it doesn't exist
        img_path = os.path.join(images_dir, file_name)
        if not os.path.exists(img_path):
            response = requests.get(img_url)
            if response.status_code == 200:
                with open(img_path, 'wb') as f:
                    f.write(response.content)
        
        if (i + 1) % 10 == 0:
            print(f"Downloaded {i + 1}/{len(img_ids)} images")

# Download train and validation sets
print("Downloading training set...")
download_coco_subset(split="train", num_samples=240)
print("\nDownloading validation set...")
download_coco_subset(split="val", num_samples=60)