# File: rl_enhancement.py
import json
import os
import time
import datetime
from typing import Dict, List, Any
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
        
        print(f"SUCCESS: RL system initialized with {len(relationships)} relationships")
    
    def run_reinforcement_learning(self, epochs=5):
        """Run RL training loop with multiple epochs"""
        if not self.rl_system:
            self.setup_reinforcement_learning()
        
        # Load relationships
        with open(self.app.relationship_json_path, 'r') as f:
            relationships = json.load(f)
        
        print(f"Starting RL Training with {epochs} epochs")
        print(f"Training on {len(relationships)} relationships")
        
        # Initialize training metrics
        training_start_time = time.time()
        training_metrics = {
            'start_time': datetime.datetime.now().isoformat(),
            'epochs': epochs,
            'total_relationships': len(relationships),
            'epoch_results': [],
            'ai_generated_images': [],
            'training_progress': []
        }
        
        all_results = []
        
        # Run multiple training episodes
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("=" * 50)
            
            epoch_start_time = time.time()
            
            try:
                # Generate AI images for this epoch
                print(f"🎨 Generating AI images for epoch {epoch + 1}...")
                ai_images = self.generate_ai_images_for_epoch(relationships, epoch)
                training_metrics['ai_generated_images'].extend(ai_images)
                
                # Run training episode
                results = self.rl_system.train_episode(relationships)
                results['epoch'] = epoch + 1
                results['ai_images_generated'] = len(ai_images)
                results['epoch_duration'] = time.time() - epoch_start_time
                
                all_results.append(results)
                training_metrics['epoch_results'].append(results)
                
                # Update progress
                progress = {
                    'epoch': epoch + 1,
                    'timestamp': datetime.datetime.now().isoformat(),
                    'detection_loss': results['detection_loss'],
                    'relationship_loss': results['relationship_loss'],
                    'reward': results['reward'],
                    'ai_images_count': len(ai_images)
                }
                training_metrics['training_progress'].append(progress)
                
                print(f"Epoch {epoch + 1} Results:")
                print(f"   Detection Loss: {results['detection_loss']:.4f}")
                print(f"   Relationship Loss: {results['relationship_loss']:.4f}")
                print(f"   Reward: {results['reward']:.4f}")
                print(f"   Exploration Rate: {results['epsilon']:.4f}")
                print(f"   AI Images Generated: {len(ai_images)}")
                print(f"   Epoch Duration: {results['epoch_duration']:.2f}s")
                
            except Exception as e:
                print(f"ERROR: Error in epoch {epoch + 1}: {e}")
                continue
        
        # Calculate final metrics
        training_duration = time.time() - training_start_time
        training_metrics['end_time'] = datetime.datetime.now().isoformat()
        training_metrics['total_duration'] = training_duration
        training_metrics['total_ai_images'] = len(training_metrics['ai_generated_images'])
        
        # Calculate average results
        if all_results:
            avg_results = {
                'detection_loss': sum(r['detection_loss'] for r in all_results) / len(all_results),
                'relationship_loss': sum(r['relationship_loss'] for r in all_results) / len(all_results),
                'reward': sum(r['reward'] for r in all_results) / len(all_results),
                'epsilon': all_results[-1]['epsilon'],  # Latest epsilon
                'total_ai_images': training_metrics['total_ai_images'],
                'training_duration': training_duration,
                'successful_epochs': len(all_results)
            }
            
            training_metrics['final_results'] = avg_results
            
            print(f"\nFinal Training Results (Average over {len(all_results)} epochs):")
            print(f"   Average Detection Loss: {avg_results['detection_loss']:.4f}")
            print(f"   Average Relationship Loss: {avg_results['relationship_loss']:.4f}")
            print(f"   Average Reward: {avg_results['reward']:.4f}")
            print(f"   Final Exploration Rate: {avg_results['epsilon']:.4f}")
            print(f"   Total AI Images Generated: {avg_results['total_ai_images']}")
            print(f"   Training Duration: {avg_results['training_duration']:.2f}s")
            
            # Save training metrics to file
            self.save_training_metrics(training_metrics)
            
            return avg_results
        else:
            print("ERROR: No successful training epochs completed")
            return None
    
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
        
        print(f"SUCCESS: Generated {len(synthetic_dataset)} synthetic images")
        return synthetic_dataset
    
    def generate_ai_images_for_epoch(self, relationships: List[Dict], epoch: int) -> List[Dict]:
        """Generate AI images for a specific training epoch"""
        print(f"🎨 Generating AI images for epoch {epoch + 1}...")
        
        ai_images = []
        for i, rel in enumerate(relationships):
            try:
                print(f"  Processing relationship {i+1}/{len(relationships)}: {rel.get('subject', 'Unknown')} {rel.get('relation', 'Unknown')} {rel.get('object', 'Unknown')}")
                
                # Generate variations for this relationship
                num_variations = 3 + (epoch % 3)  # Vary number of variations per epoch
                generated_images = self.generator.generate_from_relationship(rel, num_variations=num_variations)
                
                # Add metadata to each generated image
                for j, img_data in enumerate(generated_images):
                    img_data['epoch'] = epoch + 1
                    img_data['relationship_index'] = i
                    img_data['variation_index'] = j
                    img_data['generation_timestamp'] = datetime.datetime.now().isoformat()
                
                ai_images.extend(generated_images)
                print(f"    ✅ Generated {len(generated_images)} images")
                
            except Exception as e:
                print(f"    ❌ Error generating images for relationship {i+1}: {e}")
                continue
        
        print(f"🎨 Total AI images generated for epoch {epoch + 1}: {len(ai_images)}")
        return ai_images
    
    def save_training_metrics(self, metrics: Dict[str, Any]):
        """Save training metrics to JSON file"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = f"rl_training_metrics_{timestamp}.json"
        
        try:
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            
            print(f"📊 Training metrics saved to: {metrics_file}")
            
            # Also save a summary file for quick evaluation
            self.save_training_summary(metrics, timestamp)
            
        except Exception as e:
            print(f"❌ Error saving training metrics: {e}")
    
    def save_training_summary(self, metrics: Dict[str, Any], timestamp: str):
        """Save a summary of training results for evaluation"""
        summary_file = f"rl_training_summary_{timestamp}.json"
        
        try:
            summary = {
                'training_info': {
                    'start_time': metrics.get('start_time'),
                    'end_time': metrics.get('end_time'),
                    'total_duration': metrics.get('total_duration'),
                    'epochs': metrics.get('epochs'),
                    'total_relationships': metrics.get('total_relationships'),
                    'total_ai_images': metrics.get('total_ai_images')
                },
                'final_results': metrics.get('final_results', {}),
                'training_progress': metrics.get('training_progress', []),
                'evaluation_metrics': self.calculate_evaluation_metrics(metrics)
            }
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            print(f"📈 Training summary saved to: {summary_file}")
            
        except Exception as e:
            print(f"❌ Error saving training summary: {e}")
    
    def calculate_evaluation_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate evaluation metrics from training data"""
        try:
            epoch_results = metrics.get('epoch_results', [])
            if not epoch_results:
                return {}
            
            # Calculate improvement over epochs
            first_epoch = epoch_results[0]
            last_epoch = epoch_results[-1]
            
            detection_improvement = first_epoch['detection_loss'] - last_epoch['detection_loss']
            relationship_improvement = first_epoch['relationship_loss'] - last_epoch['relationship_loss']
            reward_improvement = last_epoch['reward'] - first_epoch['reward']
            
            # Calculate consistency (lower variance = more consistent)
            detection_losses = [r['detection_loss'] for r in epoch_results]
            relationship_losses = [r['relationship_loss'] for r in epoch_results]
            rewards = [r['reward'] for r in epoch_results]
            
            detection_consistency = 1.0 - (max(detection_losses) - min(detection_losses))
            relationship_consistency = 1.0 - (max(relationship_losses) - min(relationship_losses))
            reward_consistency = 1.0 - (max(rewards) - min(rewards))
            
            evaluation = {
                'improvement_metrics': {
                    'detection_loss_improvement': detection_improvement,
                    'relationship_loss_improvement': relationship_improvement,
                    'reward_improvement': reward_improvement
                },
                'consistency_metrics': {
                    'detection_consistency': max(0, detection_consistency),
                    'relationship_consistency': max(0, relationship_consistency),
                    'reward_consistency': max(0, reward_consistency)
                },
                'overall_score': self.calculate_overall_score(metrics)
            }
            
            return evaluation
            
        except Exception as e:
            print(f"❌ Error calculating evaluation metrics: {e}")
            return {}
    
    def calculate_overall_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall training score (0-100)"""
        try:
            final_results = metrics.get('final_results', {})
            if not final_results:
                return 0.0
            
            # Weighted score based on multiple factors
            reward_score = min(final_results.get('reward', 0) * 100, 100)  # Reward (0-1) -> (0-100)
            loss_penalty = (final_results.get('detection_loss', 1) + final_results.get('relationship_loss', 1)) * 20  # Penalty for high loss
            consistency_bonus = 10  # Bonus for completing all epochs
            
            overall_score = max(0, min(100, reward_score - loss_penalty + consistency_bonus))
            return round(overall_score, 2)
            
        except Exception as e:
            print(f"❌ Error calculating overall score: {e}")
            return 0.0