#!/usr/bin/env python3
"""
Simple test for RL Training without diffusers
"""

import sys
import os
import json

def test_rl_training():
    """Test RL training with mock generator"""
    print("Testing RL Training...")
    
    # Test relationships file
    if not os.path.exists("relationships.json"):
        print("ERROR: relationships.json not found")
        return False
    
    try:
        with open("relationships.json", 'r', encoding='utf-8') as f:
            relationships = json.load(f)
        print(f"SUCCESS: Found {len(relationships)} relationships")
    except Exception as e:
        print(f"ERROR: Cannot read relationships.json: {e}")
        return False
    
    # Test mock generator
    try:
        from PIL import Image
        import numpy as np
        
        # Create mock relationship
        test_rel = {
            'subject': 'person',
            'relation': 'holding',
            'object': 'cup'
        }
        
        # Create mock images
        mock_images = []
        for i in range(3):
            mock_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            mock_image = Image.fromarray(mock_array)
            mock_images.append({
                'image': mock_image,
                'prompt': f"person holding cup, variation {i+1}",
                'original_relationship': test_rel,
                'is_mock': True
            })
        
        print(f"SUCCESS: Created {len(mock_images)} mock images")
        
        # Test RL system
        try:
            from RL.reinforcement_learning import RelationshipReinforcementLearning
            
            # Create mock generator
            class MockGenerator:
                def generate_from_relationship(self, rel, num_variations=3):
                    return mock_images[:num_variations]
            
            generator = MockGenerator()
            rl_system = RelationshipReinforcementLearning(
                detection_model=None,
                relationship_model=None,
                generator=generator
            )
            
            # Test training episode
            print("Testing training episode...")
            results = rl_system.train_episode([test_rel])
            print(f"SUCCESS: Training episode completed")
            print(f"Results: {results}")
            
            return True
            
        except Exception as e:
            print(f"ERROR: RL system test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"ERROR: Mock generator test failed: {e}")
        return False

def main():
    """Main function"""
    print("RL Training Simple Test")
    print("=" * 30)
    
    if test_rl_training():
        print("SUCCESS: All tests passed!")
        return True
    else:
        print("FAILED: Some tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
