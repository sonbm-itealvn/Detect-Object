# File: data_augmentation.py
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import albumentations as A

class RelationshipDataAugmentation:
    def __init__(self):
        self.transform = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.RandomRotate90(p=0.3),
            A.HorizontalFlip(p=0.5),
            A.RandomScale(scale_limit=0.2, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Blur(blur_limit=3, p=0.3)
        ])
    
    def augment_relationship_data(self, image, relationship):
        # Apply geometric and photometric augmentations
        augmented = self.transform(image=image)
        
        # Create relationship variations
        relationship_variations = self.create_relationship_variations(relationship)
        
        return {
            'augmented_image': augmented['image'],
            'original_relationship': relationship,
            'relationship_variations': relationship_variations
        }
    
    def create_relationship_variations(self, relationship):
        variations = []
        
        # Spatial variations
        if relationship['relation'] == 'near':
            variations.append({
                'subject': relationship['subject'],
                'relation': 'far_from',
                'object': relationship['object']
            })
        
        # Intensity variations
        if relationship['relation'] == 'holding':
            variations.append({
                'subject': relationship['subject'],
                'relation': 'gripping',
                'object': relationship['object']
            })
        
        return variations