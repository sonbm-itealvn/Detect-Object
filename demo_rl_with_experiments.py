#!/usr/bin/env python3
"""
Demo script chạy RL Training với hệ thống experiment management mới
"""

import os
import sys
import json
import time

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_demo_relationships():
    """Tạo dữ liệu relationships demo"""
    relationships = [
        {
            "subject": "person",
            "relation": "holding",
            "object": "cup",
            "confidence": 0.85
        },
        {
            "subject": "person", 
            "relation": "sitting_on",
            "object": "chair",
            "confidence": 0.92
        },
        {
            "subject": "person",
            "relation": "looking_at",
            "object": "screen",
            "confidence": 0.78
        }
    ]
    
    # Lưu relationships.json
    with open("relationships.json", "w") as f:
        json.dump(relationships, f, indent=2)
    
    print("Created demo relationships.json")
    return relationships

def run_demo_rl_training():
    """Chạy demo RL training"""
    print("DEMO: RL TRAINING WITH EXPERIMENT MANAGEMENT")
    print("=" * 60)
    
    try:
        # 1. Tạo dữ liệu demo
        print("\n1. Setting up demo data...")
        relationships = create_demo_relationships()
        print(f"   Created {len(relationships)} relationships")
        
        # 2. Import và khởi tạo app
        print("\n2. Initializing RL system...")
        from app_console import ObjectDetectionConsoleApp
        app = ObjectDetectionConsoleApp()
        print("   App initialized successfully")
        
        # 3. Chạy RL training
        print("\n3. Running RL Training...")
        print("   This will create a new experiment with:")
        print("   - AI generated images")
        print("   - Training metrics")
        print("   - Training plots")
        print("   - Experiment tracking")
        
        # Chạy RL training với 3 epochs
        results = app.rl_enhancement.run_reinforcement_learning(epochs=3)
        
        if results:
            print("\n4. RL Training completed successfully!")
            print(f"   Final Reward: {results.get('reward', 0):.4f}")
            print(f"   Detection Loss: {results.get('detection_loss', 0):.4f}")
            print(f"   Relationship Loss: {results.get('relationship_loss', 0):.4f}")
            print(f"   Total AI Images: {results.get('total_ai_images', 0)}")
            print(f"   Training Duration: {results.get('training_duration', 0):.2f}s")
            
            # 5. Hiển thị experiment info
            print("\n5. Experiment Information:")
            experiments = app.experiment_viewer.list_all_experiments()
            if experiments:
                latest_exp = experiments[0]  # Most recent
                print(f"   Experiment ID: {latest_exp['experiment_id']}")
                print(f"   Status: {latest_exp.get('status', 'Unknown')}")
                print(f"   Directory: {latest_exp.get('path', 'Unknown')}")
                
                # Hiển thị cấu trúc thư mục
                exp_path = latest_exp.get('path', '')
                if exp_path and os.path.exists(exp_path):
                    print(f"\n   Experiment Structure:")
                    for root, dirs, files in os.walk(exp_path):
                        level = root.replace(exp_path, '').count(os.sep)
                        indent = ' ' * 2 * level
                        print(f"{indent}{os.path.basename(root)}/")
                        subindent = ' ' * 2 * (level + 1)
                        for file in files[:5]:  # Show first 5 files
                            print(f"{subindent}{file}")
                        if len(files) > 5:
                            print(f"{subindent}... and {len(files) - 5} more files")
            
            print("\n6. How to view results:")
            print("   - Run: python manage_experiments.py")
            print("   - Or use the main app menu option 7")
            print("   - Or check the experiments/ directory")
            
        else:
            print("\n4. RL Training failed!")
            
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function"""
    print("RL TRAINING DEMO WITH EXPERIMENT MANAGEMENT")
    print("=" * 60)
    print("This demo will:")
    print("1. Create demo relationship data")
    print("2. Run RL training with experiment tracking")
    print("3. Show you how to view the results")
    print()
    
    choice = input("Continue with demo? (y/n): ").strip().lower()
    if choice == 'y':
        run_demo_rl_training()
    else:
        print("Demo cancelled")

if __name__ == "__main__":
    main()
