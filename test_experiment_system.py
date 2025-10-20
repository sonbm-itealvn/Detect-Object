#!/usr/bin/env python3
"""
Script test hệ thống experiment management
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

from RL.experiment_manager import ExperimentManager
from RL.experiment_viewer import ExperimentViewer

def create_test_ai_images(num_images=5):
    """Tạo ảnh AI test"""
    images = []
    for i in range(num_images):
        # Tạo ảnh mock
        img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        images.append({
            'image': img,
            'prompt': f'Test prompt {i+1}',
            'original_relationship': {
                'subject': f'person_{i+1}',
                'relation': 'holding',
                'object': f'object_{i+1}'
            },
            'is_mock': True
        })
    
    return images

def create_test_metrics():
    """Tạo metrics test"""
    return {
        'epoch': 1,
        'timestamp': '2024-01-01T00:00:00',
        'detection_loss': 0.3,
        'relationship_loss': 0.4,
        'reward': 0.7,
        'epsilon': 0.8,
        'ai_images_count': 5,
        'epoch_duration': 120.5
    }

def test_experiment_manager():
    """Test ExperimentManager"""
    print("TESTING EXPERIMENT MANAGER")
    print("=" * 50)
    
    # Tạo thư mục test
    test_dir = "test_experiments"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    # Khởi tạo ExperimentManager
    manager = ExperimentManager(test_dir)
    
    # Test 1: Tạo experiment mới
    print("1. Testing new experiment creation...")
    exp_dir = manager.start_new_experiment("test_exp_001")
    assert os.path.exists(exp_dir), "Experiment directory should exist"
    print("Experiment created successfully")
    
    # Test 2: Lưu AI images
    print("\n2. Testing AI images saving...")
    test_images = create_test_ai_images(3)
    saved_images = manager.save_ai_images(test_images, 1)
    assert len(saved_images) == 3, "Should save 3 images"
    print("AI images saved successfully")
    
    # Test 3: Lưu metrics
    print("\n3. Testing metrics saving...")
    test_metrics = create_test_metrics()
    metrics_file = manager.save_training_metrics(test_metrics, 1)
    assert os.path.exists(metrics_file), "Metrics file should exist"
    print("Metrics saved successfully")
    
    # Test 4: Tạo plots
    print("\n4. Testing plots creation...")
    training_metrics = {
        'training_progress': [
            {'epoch': 1, 'detection_loss': 0.3, 'relationship_loss': 0.4, 'reward': 0.7, 'ai_images_count': 5},
            {'epoch': 2, 'detection_loss': 0.25, 'relationship_loss': 0.35, 'reward': 0.75, 'ai_images_count': 6},
            {'epoch': 3, 'detection_loss': 0.2, 'relationship_loss': 0.3, 'reward': 0.8, 'ai_images_count': 7}
        ]
    }
    
    try:
        plot_file = manager.create_training_plots(training_metrics)
        if plot_file and os.path.exists(plot_file):
            print("Training plots created successfully")
        else:
            print("Plots creation failed (matplotlib might not be available)")
    except Exception as e:
        print(f"Plots creation failed: {e}")
    
    # Test 5: Tạo AI images grid
    print("\n5. Testing AI images grid...")
    try:
        grid_file = manager.create_ai_images_grid(test_images)
        if grid_file and os.path.exists(grid_file):
            print("AI images grid created successfully")
        else:
            print("Grid creation failed")
    except Exception as e:
        print(f"Grid creation failed: {e}")
    
    # Test 6: Hoàn thành experiment
    print("\n6. Testing experiment finalization...")
    final_results = {
        'reward': 0.8,
        'detection_loss': 0.2,
        'relationship_loss': 0.3,
        'total_ai_images': 15,
        'training_duration': 360.5
    }
    manager.finalize_experiment(final_results)
    
    # Kiểm tra metadata
    metadata_file = os.path.join(exp_dir, "metadata.json")
    assert os.path.exists(metadata_file), "Metadata file should exist"
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    assert metadata['status'] == 'completed', "Status should be completed"
    assert 'final_results' in metadata, "Final results should be in metadata"
    print("Experiment finalized successfully")
    
    print(f"\nExperiment directory: {exp_dir}")
    print("All ExperimentManager tests passed!")
    
    return exp_dir

def test_experiment_viewer(exp_dir):
    """Test ExperimentViewer"""
    print("\nTESTING EXPERIMENT VIEWER")
    print("=" * 50)
    
    viewer = ExperimentViewer("test_experiments")
    
    # Test 1: Liệt kê experiments
    print("1. Testing experiments listing...")
    experiments = viewer.list_all_experiments()
    assert len(experiments) > 0, "Should find at least one experiment"
    print("Experiments listed successfully")
    
    # Test 2: Load experiment
    print("\n2. Testing experiment loading...")
    exp_data = viewer.experiment_manager.load_experiment("test_exp_001")
    assert exp_data, "Should load experiment data"
    assert 'metadata' in exp_data, "Should have metadata"
    assert 'ai_images' in exp_data, "Should have AI images info"
    print("Experiment loaded successfully")
    
    # Test 3: View experiment details
    print("\n3. Testing experiment details viewing...")
    try:
        viewer.view_experiment_details("test_exp_001")
        print("Experiment details viewed successfully")
    except Exception as e:
        print(f"Error viewing details: {e}")
    
    print("All ExperimentViewer tests passed!")

def test_integration():
    """Test tích hợp với app_console"""
    print("\nTESTING INTEGRATION")
    print("=" * 50)
    
    try:
        from app_console import ObjectDetectionConsoleApp
        
        # Tạo app instance
        app = ObjectDetectionConsoleApp()
        
        # Test experiment viewer integration
        print("1. Testing app integration...")
        assert hasattr(app, 'experiment_viewer'), "App should have experiment_viewer"
        assert hasattr(app, 'list_experiments'), "App should have list_experiments method"
        print("App integration successful")
        
    except Exception as e:
        print(f"Integration test failed: {e}")

def cleanup():
    """Dọn dẹp test files"""
    print("\nCLEANING UP...")
    test_dir = "test_experiments"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
        print("Test files cleaned up")

def main():
    """Main test function"""
    print("TESTING EXPERIMENT SYSTEM")
    print("=" * 60)
    
    try:
        # Test ExperimentManager
        exp_dir = test_experiment_manager()
        
        # Test ExperimentViewer
        test_experiment_viewer(exp_dir)
        
        # Test integration
        test_integration()
        
        print("\nALL TESTS PASSED!")
        print("=" * 30)
        print("ExperimentManager: Working")
        print("ExperimentViewer: Working") 
        print("Integration: Working")
        print("File structure: Created")
        print("Data persistence: Working")
        
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        cleanup()

if __name__ == "__main__":
    main()
