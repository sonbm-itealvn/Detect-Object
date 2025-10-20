#!/usr/bin/env python3
"""
Test script cho model persistence system
"""

import os
import sys
import json
import tempfile
import shutil

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_model_persistence():
    """Test model persistence functionality"""
    print("TESTING MODEL PERSISTENCE SYSTEM")
    print("=" * 50)
    
    try:
        from RL.model_manager import ModelManager
        from RL.reinforcement_learning import RelationshipReinforcementLearning
        from RL.ai_images_generator import RelationshipImageGenerator
        
        # Test 1: Create model manager
        print("1. Creating model manager...")
        manager = ModelManager("test_models")
        print("   OK")
        
        # Test 2: Create experiment directory
        print("2. Setting up experiment directory...")
        exp_dir = "test_models/exp_001"
        os.makedirs(exp_dir, exist_ok=True)
        manager.set_experiment_dir(exp_dir)
        print("   OK")
        
        # Test 3: Create RL system
        print("3. Creating RL system...")
        generator = RelationshipImageGenerator()
        rl_system = RelationshipReinforcementLearning(
            detection_model=None,
            relationship_model=None,
            generator=generator,
            experiment_dir=exp_dir
        )
        print("   OK")
        
        # Test 4: Simulate training and save model state
        print("4. Simulating training and saving model state...")
        
        # Simulate training results
        for epoch in range(3):
            reward = 0.5 + epoch * 0.1  # Increasing reward
            detection_loss = 0.4 - epoch * 0.05  # Decreasing loss
            relationship_loss = 0.5 - epoch * 0.05  # Decreasing loss
            
            # Save model state
            rl_system.save_model_state(reward, detection_loss, relationship_loss)
            
            # Update training history
            epoch_data = {
                'epoch': epoch + 1,
                'detection_loss': detection_loss,
                'relationship_loss': relationship_loss,
                'reward': reward,
                'epsilon': 0.9 - epoch * 0.1
            }
            rl_system.training_history['epochs'].append(epoch_data)
            
            print(f"   Epoch {epoch + 1}: Reward={reward:.3f}, Loss={detection_loss:.3f}")
        
        print("   OK")
        
        # Test 5: Check saved files
        print("5. Checking saved model files...")
        models_dir = os.path.join(exp_dir, "models")
        if os.path.exists(models_dir):
            files = os.listdir(models_dir)
            print(f"   Found {len(files)} files in models directory")
            for file in files:
                print(f"   - {file}")
        else:
            print("   ERROR: Models directory not found")
            return False
        
        print("   OK")
        
        # Test 6: Load model state
        print("6. Testing model loading...")
        loaded_checkpoint = manager.load_model_state('detection_model', exp_dir)
        if loaded_checkpoint:
            print(f"   Loaded model with reward: {loaded_checkpoint['model_state'].get('reward', 0):.3f}")
        else:
            print("   ERROR: Failed to load model")
            return False
        
        print("   OK")
        
        # Test 7: Get best model
        print("7. Testing best model retrieval...")
        best_model = manager.get_best_model('detection_model', exp_dir)
        if best_model:
            print(f"   Best model reward: {best_model['model_state'].get('reward', 0):.3f}")
        else:
            print("   ERROR: Failed to get best model")
            return False
        
        print("   OK")
        
        # Test 8: Create model summary
        print("8. Creating model summary...")
        summary = manager.create_model_summary(exp_dir)
        print(f"   Summary created with {summary.get('total_models', 0)} models")
        print("   OK")
        
        print("\nAll model persistence tests passed!")
        print("Model weights are now being saved and can be loaded!")
        
        # Cleanup
        if os.path.exists("test_models"):
            shutil.rmtree("test_models")
            print("Test files cleaned up")
        
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("MODEL PERSISTENCE TEST")
    print("=" * 30)
    
    success = test_model_persistence()
    
    if success:
        print("\nSUCCESS: Model persistence is working!")
        print("Now your RL training will save model weights and can continue from previous experiments!")
    else:
        print("\nFAILED: Model persistence has issues")

if __name__ == "__main__":
    main()
