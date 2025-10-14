#!/usr/bin/env python3
"""
Simple debug script for RL Training
"""

import sys
import os
import json

def test_basic_imports():
    """Test basic imports"""
    print("Testing basic imports...")
    
    try:
        import torch
        print("SUCCESS: torch imported")
    except Exception as e:
        print(f"ERROR: torch import failed: {e}")
        return False
    
    try:
        from diffusers import StableDiffusionPipeline
        print("SUCCESS: diffusers imported")
    except Exception as e:
        print(f"ERROR: diffusers import failed: {e}")
        return False
    
    return True

def test_relationships_file():
    """Test relationships file"""
    print("Testing relationships file...")
    
    if not os.path.exists("relationships.json"):
        print("ERROR: relationships.json not found")
        return False
    
    try:
        with open("relationships.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"SUCCESS: Found {len(data)} relationships")
        return True
    except Exception as e:
        print(f"ERROR: Cannot read relationships.json: {e}")
        return False

def test_mock_generator():
    """Test mock generator without Stable Diffusion"""
    print("Testing mock generator...")
    
    try:
        from PIL import Image
        import numpy as np
        
        # Create mock image
        mock_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        mock_image = Image.fromarray(mock_array)
        print("SUCCESS: Mock image created")
        return True
    except Exception as e:
        print(f"ERROR: Mock generator failed: {e}")
        return False

def main():
    """Main function"""
    print("RL Training Debug")
    print("=" * 30)
    
    if not test_basic_imports():
        print("FAILED: Basic imports")
        return False
    
    if not test_relationships_file():
        print("FAILED: Relationships file")
        return False
    
    if not test_mock_generator():
        print("FAILED: Mock generator")
        return False
    
    print("SUCCESS: All basic tests passed")
    return True

if __name__ == "__main__":
    main()
