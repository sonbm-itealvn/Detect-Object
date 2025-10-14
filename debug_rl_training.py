#!/usr/bin/env python3
"""
Debug script để kiểm tra RL Training
"""

import sys
import os
import json

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        from RL.ai_images_generator import RelationshipImageGenerator
        print("SUCCESS: RelationshipImageGenerator imported successfully")
    except Exception as e:
        print(f"ERROR: Error importing RelationshipImageGenerator: {e}")
        return False
    
    try:
        from RL.reinforcement_learning import RelationshipReinforcementLearning
        print("SUCCESS: RelationshipReinforcementLearning imported successfully")
    except Exception as e:
        print(f"ERROR: Error importing RelationshipReinforcementLearning: {e}")
        return False
    
    try:
        from RL.rl_enhancement import AppReinforcementLearning
        print("SUCCESS: AppReinforcementLearning imported successfully")
    except Exception as e:
        print(f"ERROR: Error importing AppReinforcementLearning: {e}")
        return False
    
    return True

def test_generator():
    """Test if generator can be initialized"""
    print("\n🎨 Testing AI Image Generator...")
    
    try:
        from RL.ai_images_generator import RelationshipImageGenerator
        
        print("🔄 Initializing generator...")
        generator = RelationshipImageGenerator()
        print("✅ Generator initialized successfully")
        
        # Test mock generation
        test_relationship = {
            'subject': 'person',
            'relation': 'holding',
            'object': 'cup'
        }
        
        print("🔄 Testing mock generation...")
        images = generator.generate_from_relationship(test_relationship, num_variations=2)
        print(f"✅ Generated {len(images)} mock images")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing generator: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_relationships_file():
    """Test if relationships.json exists and is valid"""
    print("\n📄 Testing relationships file...")
    
    relationships_path = "relationships.json"
    
    if not os.path.exists(relationships_path):
        print(f"❌ File {relationships_path} not found")
        return False
    
    try:
        with open(relationships_path, 'r', encoding='utf-8') as f:
            relationships = json.load(f)
        
        print(f"✅ File is valid JSON with {len(relationships)} relationships")
        
        if relationships:
            sample = relationships[0]
            print(f"📋 Sample relationship: {sample}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading relationships file: {e}")
        return False

def test_rl_system():
    """Test if RL system can be initialized"""
    print("\n🧠 Testing RL System...")
    
    try:
        from RL.ai_images_generator import RelationshipImageGenerator
        from RL.reinforcement_learning import RelationshipReinforcementLearning
        
        # Create mock app
        class MockApp:
            def __init__(self):
                self.relationship_json_path = "relationships.json"
        
        mock_app = MockApp()
        
        print("🔄 Initializing RL system...")
        generator = RelationshipImageGenerator()
        rl_system = RelationshipReinforcementLearning(
            detection_model=None,
            relationship_model=None,
            generator=generator
        )
        
        print("✅ RL system initialized successfully")
        
        # Test with mock relationships
        test_relationships = [
            {'subject': 'person', 'relation': 'holding', 'object': 'cup'},
            {'subject': 'person', 'relation': 'sitting_on', 'object': 'chair'}
        ]
        
        print("🔄 Testing training episode...")
        results = rl_system.train_episode(test_relationships)
        print(f"✅ Training episode completed with results: {results}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing RL system: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main debug function"""
    print("RL Training Debug Script")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed")
        return False
    
    # Test relationships file
    if not test_relationships_file():
        print("\n❌ Relationships file test failed")
        return False
    
    # Test generator
    if not test_generator():
        print("\n❌ Generator test failed")
        return False
    
    # Test RL system
    if not test_rl_system():
        print("\n❌ RL system test failed")
        return False
    
    print("\n🎉 All tests passed! RL Training should work now.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
