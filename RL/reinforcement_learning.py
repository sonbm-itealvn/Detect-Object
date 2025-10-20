# File: reinforcement_learning.py
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from collections import deque
import random
import os
import datetime
from typing import Any, Dict, List
from RL.model_manager import ModelManager

class RelationshipReinforcementLearning:
    def __init__(self, detection_model, relationship_model, generator, experiment_dir=None):
        self.detection_model = detection_model  # Can be None initially
        self.relationship_model = relationship_model  # Can be None initially
        self.generator = generator
        self.memory = deque(maxlen=10000)
        self.epsilon = 0.9  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # Model management
        self.model_manager = ModelManager()
        if experiment_dir:
            self.model_manager.set_experiment_dir(experiment_dir)
        
        # Training history
        self.training_history = {
            'epochs': [],
            'best_reward': float('-inf'),
            'best_epoch': 0
        }
        
    def train_episode(self, original_relationships, synthetic_data=None):
        print(f"Starting training episode with {len(original_relationships)} relationships")
        
        # 1. Use provided synthetic data or generate new if none provided
        if synthetic_data is None:
            print("Step 1: Generating synthetic data...")
            synthetic_data = []
            for i, rel in enumerate(original_relationships):
                print(f"  Processing relationship {i+1}/{len(original_relationships)}: {rel.get('subject', 'Unknown')} {rel.get('relation', 'Unknown')} {rel.get('object', 'Unknown')}")
                try:
                    generated_images = self.generator.generate_from_relationship(rel, num_variations=3)
                    synthetic_data.extend(generated_images)
                    print(f"    SUCCESS: Generated {len(generated_images)} images")
                except Exception as e:
                    print(f"    ERROR: Error generating images for relationship {i+1}: {e}")
                    continue
            print(f"Total synthetic data generated: {len(synthetic_data)} images")
        else:
            print(f"Step 1: Using provided synthetic data: {len(synthetic_data)} images")
        
        # 2. Train detection model
        print("Step 2: 🧠 Training detection model...")
        detection_loss = self.train_detection_model(synthetic_data)
        print(f"    ✅ Detection loss: {detection_loss:.4f}")
        
        # 3. Train relationship model
        print("Step 3: 🧠 Training relationship model...")
        relationship_loss = self.train_relationship_model(synthetic_data)
        print(f"    ✅ Relationship loss: {relationship_loss:.4f}")
        
        # 4. Calculate reward
        print("Step 4: 📊 Calculating reward...")
        reward = self.calculate_reward(synthetic_data, original_relationships)
        print(f"    ✅ Reward: {reward:.4f}")
        
        # 5. Update exploration rate
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        print(f"    Exploration rate: {self.epsilon:.4f}")
        
        # 6. Save model state if this is a good result
        if reward > self.training_history['best_reward']:
            self.training_history['best_reward'] = reward
            self.training_history['best_epoch'] = len(self.training_history['epochs']) + 1
            self.save_model_state(reward, detection_loss, relationship_loss)
        
        # 7. Update training history
        epoch_data = {
            'epoch': len(self.training_history['epochs']) + 1,
            'detection_loss': detection_loss,
            'relationship_loss': relationship_loss,
            'reward': reward,
            'epsilon': self.epsilon,
            'timestamp': datetime.datetime.now().isoformat()
        }
        self.training_history['epochs'].append(epoch_data)
        
        print("SUCCESS: Training episode completed!")
        
        experience_batch = self._build_experience_batch(
            original_relationships=original_relationships,
            synthetic_data=synthetic_data,
            epoch_index=len(self.training_history['epochs']) + 1,
            reward=reward,
            detection_loss=detection_loss,
            relationship_loss=relationship_loss,
        )
        
        return {
            'detection_loss': detection_loss,
            'relationship_loss': relationship_loss,
            'reward': reward,
            'epsilon': self.epsilon,
            'experience_batch': experience_batch
        }
    
    def calculate_reward(self, synthetic_data, original_relationships):
        # Accuracy reward
        accuracy_reward = self.calculate_accuracy_reward(synthetic_data, original_relationships)
        
        # Diversity reward
        diversity_reward = self.calculate_diversity_reward(synthetic_data)
        
        # Consistency reward
        consistency_reward = self.calculate_consistency_reward(synthetic_data)
        
        total_reward = 0.4 * accuracy_reward + 0.3 * diversity_reward + 0.3 * consistency_reward
        return total_reward
    
    def calculate_accuracy_reward(self, synthetic_data, original_relationships):
        # Simulate detection and relationship prediction
        correct_predictions = 0
        total_predictions = 0
        
        for data in synthetic_data:
            # Mock prediction (replace with actual model inference)
            predicted_relationships = self.predict_relationships(data['image'])
            
            # Compare with original relationships
            for pred_rel in predicted_relationships:
                for orig_rel in original_relationships:
                    if self.relationship_similarity(pred_rel, orig_rel) > 0.7:
                        correct_predictions += 1
                    total_predictions += 1
        
        return correct_predictions / max(total_predictions, 1)
    
    def relationship_similarity(self, rel1, rel2):
        # Calculate similarity between two relationships
        subject_sim = 1.0 if rel1['subject'] == rel2['subject'] else 0.0
        relation_sim = 1.0 if rel1['relation'] == rel2['relation'] else 0.0
        object_sim = 1.0 if rel1['object'] == rel2['object'] else 0.0
        
        return (subject_sim + relation_sim + object_sim) / 3.0
    
    def train_detection_model(self, synthetic_data):
        """Train detection model with synthetic data"""
        if self.detection_model is None:
            print("WARNING: Detection model not available, using simulated training")
            return random.uniform(0.1, 0.4)  # Simulated loss
        
        # TODO: Implement actual detection model training
        # This would involve:
        # 1. Load YOLO model
        # 2. Train on synthetic data
        # 3. Calculate loss
        return random.uniform(0.1, 0.4)  # Simulated loss
    
    def train_relationship_model(self, synthetic_data):
        """Train relationship model with synthetic data"""
        if self.relationship_model is None:
            print("WARNING: Relationship model not available, using simulated training")
            return random.uniform(0.15, 0.5)  # Simulated loss
        
        # TODO: Implement actual relationship model training
        # This would involve:
        # 1. Load RelTR model
        # 2. Train on synthetic data
        # 3. Calculate loss
        return random.uniform(0.15, 0.5)  # Simulated loss
    
    def predict_relationships(self, image):
        """Predict relationships from image (mock implementation)"""
        # Mock prediction - replace with actual model inference
        mock_relationships = [
            {'subject': 'person', 'relation': 'holding', 'object': 'cup'},
            {'subject': 'person', 'relation': 'sitting_on', 'object': 'chair'}
        ]
        return mock_relationships
    
    def calculate_diversity_reward(self, synthetic_data):
        """Calculate diversity reward based on synthetic data variety"""
        if not synthetic_data:
            return 0.0
        
        # Count unique relationship types
        unique_relations = set()
        for data in synthetic_data:
            if 'original_relationship' in data:
                rel = data['original_relationship']
                unique_relations.add(rel.get('relation', ''))
        
        # Diversity reward based on number of unique relations
        diversity_score = min(len(unique_relations) / 10.0, 1.0)  # Normalize to [0,1]
        return diversity_score
    
    def calculate_consistency_reward(self, synthetic_data):
        """Calculate consistency reward based on synthetic data consistency"""
        if not synthetic_data:
            return 0.0
        
        # Mock consistency calculation
        # In real implementation, this would check consistency between
        # synthetic data and original relationships
        consistency_score = random.uniform(0.6, 0.9)
        return consistency_score
    
    def save_model_state(self, reward, detection_loss, relationship_loss):
        """Lưu trạng thái model khi có kết quả tốt"""
        if not self.model_manager.current_experiment_dir:
            print("WARNING: No experiment directory set, cannot save model state")
            return
        
        try:
            # Tạo model state (trong thực tế sẽ là actual model weights)
            model_state = {
                'epsilon': self.epsilon,
                'reward': reward,
                'detection_loss': detection_loss,
                'relationship_loss': relationship_loss,
                'training_step': len(self.training_history['epochs']),
                'model_weights': self.create_mock_model_weights(),  # Mock weights
                'optimizer_state': self.create_mock_optimizer_state()  # Mock optimizer state
            }
            
            # Lưu detection model state
            self.model_manager.save_model_state(
                'detection_model',
                model_state,
                epoch=len(self.training_history['epochs']),
                metadata={
                    'reward': reward,
                    'detection_loss': detection_loss,
                    'relationship_loss': relationship_loss,
                    'epsilon': self.epsilon
                }
            )
            
            # Lưu relationship model state
            self.model_manager.save_model_state(
                'relationship_model',
                model_state,
                epoch=len(self.training_history['epochs']),
                metadata={
                    'reward': reward,
                    'detection_loss': detection_loss,
                    'relationship_loss': relationship_loss,
                    'epsilon': self.epsilon
                }
            )
            
            # Lưu training history
            self.model_manager.save_training_history(self.training_history)
            
            print(f"Saved model state (reward: {reward:.4f})")
            
        except Exception as e:
            print(f"ERROR saving model state: {e}")
    
    def load_model_state(self, model_name='detection_model', experiment_dir=None):
        """Load trạng thái model từ checkpoint"""
        try:
            checkpoint = self.model_manager.load_model_state(model_name, experiment_dir)
            if checkpoint:
                # Restore model state
                self.epsilon = checkpoint['model_state'].get('epsilon', self.epsilon)
                
                # Restore training history
                if 'training_history' in checkpoint:
                    self.training_history = checkpoint['training_history']
                
                print(f"Loaded model state from {model_name}")
                return True
            else:
                print(f"No checkpoint found for {model_name}")
                return False
                
        except Exception as e:
            print(f"ERROR loading model state: {e}")
            return False
    
    def create_mock_model_weights(self):
        """Tạo mock model weights (trong thực tế sẽ là actual weights)"""
        return {
            'layer1_weight': np.random.randn(10, 10).tolist(),
            'layer1_bias': np.random.randn(10).tolist(),
            'layer2_weight': np.random.randn(5, 10).tolist(),
            'layer2_bias': np.random.randn(5).tolist(),
            'timestamp': datetime.datetime.now().isoformat()
        }
    
    def create_mock_optimizer_state(self):
        """Tạo mock optimizer state"""
        return {
            'step': len(self.training_history['epochs']),
            'learning_rate': 0.001,
            'momentum': 0.9,
            'timestamp': datetime.datetime.now().isoformat()
        }
    
    def get_best_model(self, metric='reward'):
        """Lấy model tốt nhất"""
        return self.model_manager.get_best_model('detection_model', metric=metric)
    
    def continue_training(self, experiment_dir):
        """Tiếp tục training từ experiment trước đó"""
        self.model_manager.set_experiment_dir(experiment_dir)
        
        # Load best model từ experiment trước
        best_model = self.get_best_model()
        if best_model:
            print(f"Continuing training from best model (reward: {best_model['model_state'].get('reward', 0):.4f})")
            return True
        else:
            print("No previous model found, starting fresh training")
            return False

    # ------------------------------------------------------------------ #
    # Experience helpers
    # ------------------------------------------------------------------ #
    def _build_experience_batch(
        self,
        original_relationships: List[Dict[str, Any]],
        synthetic_data: List[Dict[str, Any]],
        epoch_index: int,
        reward: float,
        detection_loss: float,
        relationship_loss: float,
    ) -> List[Dict[str, Any]]:
        """Create a batch of serialized experiences for replay buffer storage."""
        if not synthetic_data:
            summary_state = {
                'epoch': epoch_index,
                'epsilon': self.epsilon,
                'relationships_observed': len(original_relationships),
            }
            return [{
                'state': summary_state,
                'action': 'train_models',
                'reward': reward,
                'next_state': {**summary_state},
                'done': True,
                'metadata': {
                    'detection_loss': detection_loss,
                    'relationship_loss': relationship_loss,
                    'timestamp': datetime.datetime.now().isoformat(),
                }
            }]

        per_step_reward = reward / max(len(synthetic_data), 1)
        batch: List[Dict[str, Any]] = []

        for idx, data in enumerate(synthetic_data):
            relationship_info = self._extract_relationship_info(data)
            state = {
                'epoch': epoch_index,
                'step': idx,
                'epsilon': self.epsilon,
                'relationship': relationship_info,
            }
            next_state = {
                'epoch': epoch_index,
                'step': idx + 1,
                'epsilon': max(self.epsilon * self.epsilon_decay, self.epsilon_min),
                'relationship': relationship_info,
            }

            batch.append({
                'state': state,
                'action': 'train_relationship_models',
                'reward': per_step_reward,
                'next_state': next_state,
                'done': False,
                'metadata': {
                    'prompt': data.get('prompt'),
                    'is_mock': data.get('is_mock', False),
                    'detection_loss': detection_loss,
                    'relationship_loss': relationship_loss,
                    'generated_at': data.get('generation_timestamp'),
                }
            })

        if batch:
            batch[-1]['done'] = True
            batch[-1]['metadata']['epoch_reward'] = reward

        return batch

    @staticmethod
    def _extract_relationship_info(data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract a serializable snapshot of the relationship sample."""
        relationship = data.get('original_relationship') or {}
        cleaned_relationship = {
            key: value
            for key, value in relationship.items()
            if key not in {'image', 'image_data'}
        }

        return {
            'subject': cleaned_relationship.get('subject'),
            'relation': cleaned_relationship.get('relation'),
            'object': cleaned_relationship.get('object'),
            'similarity': cleaned_relationship.get('visual_similarity'),
            'epoch': data.get('epoch'),
            'variation_index': data.get('variation_index'),
            'relationship_index': data.get('relationship_index'),
        }
