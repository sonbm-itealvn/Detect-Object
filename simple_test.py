#!/usr/bin/env python3
"""
Simple test for experiment system
"""

import os
import sys
import json
import tempfile
import shutil
from PIL import Image
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_basic_functionality():
    """Test basic functionality without emoji"""
    print("Testing Experiment System")
    print("=" * 50)
    
    try:
        from RL.experiment_manager import ExperimentManager
        
        # Test 1: Create experiment manager
        print("1. Creating experiment manager...")
        manager = ExperimentManager("test_experiments")
        print("   OK")
        
        # Test 2: Start new experiment
        print("2. Starting new experiment...")
        exp_dir = manager.start_new_experiment("test_exp_001")
        print(f"   Experiment directory: {exp_dir}")
        print("   OK")
        
        # Test 3: Create test AI images
        print("3. Creating test AI images...")
        test_images = []
        for i in range(3):
            img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            test_images.append({
                'image': img,
                'prompt': f'Test prompt {i+1}',
                'original_relationship': {
                    'subject': f'person_{i+1}',
                    'relation': 'holding',
                    'object': f'object_{i+1}'
                },
                'is_mock': True
            })
        print("   OK")
        
        # Test 4: Save AI images
        print("4. Saving AI images...")
        saved_images = manager.save_ai_images(test_images, 1)
        print(f"   Saved {len(saved_images)} images")
        print("   OK")
        
        # Test 5: Save metrics
        print("5. Saving metrics...")
        test_metrics = {
            'epoch': 1,
            'detection_loss': 0.3,
            'relationship_loss': 0.4,
            'reward': 0.7,
            'ai_images_count': 3
        }
        metrics_file = manager.save_training_metrics(test_metrics, 1)
        print(f"   Metrics saved to: {metrics_file}")
        print("   OK")
        
        # Test 6: Finalize experiment
        print("6. Finalizing experiment...")
        final_results = {
            'reward': 0.8,
            'detection_loss': 0.2,
            'relationship_loss': 0.3,
            'total_ai_images': 3
        }
        manager.finalize_experiment(final_results)
        print("   OK")
        
        # Test 7: Check files created
        print("7. Checking created files...")
        assert os.path.exists(exp_dir), "Experiment directory should exist"
        assert os.path.exists(os.path.join(exp_dir, "metadata.json")), "Metadata should exist"
        assert os.path.exists(os.path.join(exp_dir, "ai_images")), "AI images directory should exist"
        assert os.path.exists(os.path.join(exp_dir, "metrics")), "Metrics directory should exist"
        print("   OK")
        
        print("\nAll tests passed!")
        print("Experiment system is working correctly")
        
        # Cleanup
        if os.path.exists("test_experiments"):
            shutil.rmtree("test_experiments")
            print("Test files cleaned up")
        
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_functionality()
    if success:
        print("\nSUCCESS: Experiment system is ready to use!")
    else:
        print("\nFAILED: Experiment system has issues")
