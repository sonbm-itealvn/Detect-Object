# File: rl_enhancement.py
import json
import os
from RL.ai_images_generator import RelationshipImageGenerator
from RL.data_augmentation import RelationshipDataAugmentation
from RL.reinforcement_learning import RelationshipReinforcementLearning

class AppReinforcementLearning:
    def __init__(self, app_instance):
        self.app = app_instance
        self.generator = RelationshipImageGenerator()
        self.augmentation = RelationshipDataAugmentation()
        self.rl_system = None
        
    def setup_reinforcement_learning(self):
        """Setup RL system after initial detection"""
        # Load current relationships
        with open(self.app.relationship_json_path, 'r') as f:
            relationships = json.load(f)
        
        if not relationships:
            print("❌ No relationships found for RL training")
            return
        
        # Initialize RL system
        # Note: These models are loaded dynamically during pipeline execution
        # detection_model = YOLO model (loaded in detect_objects.py)
        # relationship_model = RelTR model (loaded in boundingbox_objects.py)
        self.rl_system = RelationshipReinforcementLearning(
            detection_model=None,  # Will be loaded from YOLO pipeline
            relationship_model=None,  # Will be loaded from RelTR pipeline
            generator=self.generator
        )
        
        print(f"✅ RL system initialized with {len(relationships)} relationships")
    
    def run_reinforcement_learning(self):
        """Run RL training loop"""
        if not self.rl_system:
            self.setup_reinforcement_learning()
        
        # Load relationships
        with open(self.app.relationship_json_path, 'r') as f:
            relationships = json.load(f)
        
        # Run training episode
        results = self.rl_system.train_episode(relationships)
        
        print(f"🎯 RL Training Results:")
        print(f"   Detection Loss: {results['detection_loss']:.4f}")
        print(f"   Relationship Loss: {results['relationship_loss']:.4f}")
        print(f"   Reward: {results['reward']:.4f}")
        print(f"   Exploration Rate: {results['epsilon']:.4f}")
        
        return results
    
    def generate_synthetic_dataset(self, num_variations=5):
        """Generate synthetic dataset from current relationships"""
        with open(self.app.relationship_json_path, 'r') as f:
            relationships = json.load(f)
        
        synthetic_dataset = []
        for rel in relationships:
            generated_images = self.generator.generate_from_relationship(
                rel, num_variations=num_variations
            )
            synthetic_dataset.extend(generated_images)
        
        # Save synthetic dataset
        synthetic_path = "synthetic_relationships.json"
        with open(synthetic_path, 'w') as f:
            json.dump(synthetic_dataset, f, indent=2)
        
        print(f"✅ Generated {len(synthetic_dataset)} synthetic images")
        return synthetic_dataset