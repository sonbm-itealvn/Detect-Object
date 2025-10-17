# File: training_evaluator.py
import json
import os
import glob
from typing import Dict, List, Any
import datetime

class TrainingEvaluator:
    def __init__(self):
        self.metrics_files = []
        self.summary_files = []
        
    def find_training_files(self):
        """Find all training metrics and summary files"""
        # Find metrics files
        metrics_pattern = "rl_training_metrics_*.json"
        self.metrics_files = glob.glob(metrics_pattern)
        
        # Find summary files
        summary_pattern = "rl_training_summary_*.json"
        self.summary_files = glob.glob(summary_pattern)
        
        print(f"Found {len(self.metrics_files)} metrics files")
        print(f"Found {len(self.summary_files)} summary files")
        
        return self.metrics_files, self.summary_files
    
    def load_training_data(self, file_path: str) -> Dict[str, Any]:
        """Load training data from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
            return {}
    
    def evaluate_training_session(self, metrics_file: str) -> Dict[str, Any]:
        """Evaluate a single training session"""
        print(f"\n📊 Evaluating training session: {metrics_file}")
        
        metrics = self.load_training_data(metrics_file)
        if not metrics:
            return {}
        
        evaluation = {
            'file_name': os.path.basename(metrics_file),
            'training_info': metrics.get('training_info', {}),
            'final_results': metrics.get('final_results', {}),
            'evaluation_metrics': metrics.get('evaluation_metrics', {}),
            'training_progress': metrics.get('training_progress', []),
            'ai_images_info': self.analyze_ai_images(metrics),
            'performance_analysis': self.analyze_performance(metrics),
            'recommendations': self.generate_recommendations(metrics)
        }
        
        return evaluation
    
    def analyze_ai_images(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze AI image generation performance"""
        ai_images = metrics.get('ai_generated_images', [])
        
        if not ai_images:
            return {'total_images': 0, 'analysis': 'No AI images generated'}
        
        # Count images by epoch
        epoch_counts = {}
        for img in ai_images:
            epoch = img.get('epoch', 0)
            epoch_counts[epoch] = epoch_counts.get(epoch, 0) + 1
        
        # Analyze image quality (mock analysis)
        mock_images = sum(1 for img in ai_images if img.get('is_mock', False))
        real_images = len(ai_images) - mock_images
        
        analysis = {
            'total_images': len(ai_images),
            'real_images': real_images,
            'mock_images': mock_images,
            'images_by_epoch': epoch_counts,
            'quality_ratio': real_images / len(ai_images) if ai_images else 0,
            'average_images_per_epoch': len(ai_images) / len(epoch_counts) if epoch_counts else 0
        }
        
        return analysis
    
    def analyze_performance(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze training performance"""
        progress = metrics.get('training_progress', [])
        if not progress:
            return {'analysis': 'No training progress data available'}
        
        # Calculate performance trends
        detection_losses = [p['detection_loss'] for p in progress]
        relationship_losses = [p['relationship_loss'] for p in progress]
        rewards = [p['reward'] for p in progress]
        
        # Trend analysis
        detection_trend = self.calculate_trend(detection_losses)
        relationship_trend = self.calculate_trend(relationship_losses)
        reward_trend = self.calculate_trend(rewards)
        
        # Stability analysis
        detection_stability = self.calculate_stability(detection_losses)
        relationship_stability = self.calculate_stability(relationship_losses)
        reward_stability = self.calculate_stability(rewards)
        
        analysis = {
            'detection_loss': {
                'trend': detection_trend,
                'stability': detection_stability,
                'final_value': detection_losses[-1] if detection_losses else 0,
                'improvement': detection_losses[0] - detection_losses[-1] if len(detection_losses) > 1 else 0
            },
            'relationship_loss': {
                'trend': relationship_trend,
                'stability': relationship_stability,
                'final_value': relationship_losses[-1] if relationship_losses else 0,
                'improvement': relationship_losses[0] - relationship_losses[-1] if len(relationship_losses) > 1 else 0
            },
            'reward': {
                'trend': reward_trend,
                'stability': reward_stability,
                'final_value': rewards[-1] if rewards else 0,
                'improvement': rewards[-1] - rewards[0] if len(rewards) > 1 else 0
            }
        }
        
        return analysis
    
    def calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 2:
            return 'insufficient_data'
        
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        
        if second_avg > first_avg * 1.05:
            return 'increasing'
        elif second_avg < first_avg * 0.95:
            return 'decreasing'
        else:
            return 'stable'
    
    def calculate_stability(self, values: List[float]) -> float:
        """Calculate stability score (0-1, higher is more stable)"""
        if len(values) < 2:
            return 0.0
        
        variance = sum((x - sum(values)/len(values))**2 for x in values) / len(values)
        max_variance = max(values) - min(values)
        
        if max_variance == 0:
            return 1.0
        
        stability = 1.0 - (variance / max_variance)
        return max(0.0, min(1.0, stability))
    
    def generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on training results"""
        recommendations = []
        
        final_results = metrics.get('final_results', {})
        evaluation_metrics = metrics.get('evaluation_metrics', {})
        
        # Check reward performance
        reward = final_results.get('reward', 0)
        if reward < 0.3:
            recommendations.append("🔴 Low reward score - consider increasing training epochs or adjusting learning parameters")
        elif reward > 0.7:
            recommendations.append("🟢 Good reward score - training is performing well")
        
        # Check loss values
        detection_loss = final_results.get('detection_loss', 1)
        relationship_loss = final_results.get('relationship_loss', 1)
        
        if detection_loss > 0.5:
            recommendations.append("🔴 High detection loss - consider more detection training data or model tuning")
        
        if relationship_loss > 0.5:
            recommendations.append("🔴 High relationship loss - consider more relationship training data or model tuning")
        
        # Check AI image generation
        ai_images_info = self.analyze_ai_images(metrics)
        if ai_images_info.get('total_images', 0) < 10:
            recommendations.append("🟡 Low AI image generation - consider increasing variations per relationship")
        
        if ai_images_info.get('quality_ratio', 0) < 0.5:
            recommendations.append("🟡 Low AI image quality - consider improving image generation parameters")
        
        # Check training duration
        duration = final_results.get('training_duration', 0)
        if duration > 3600:  # More than 1 hour
            recommendations.append("🟡 Long training duration - consider optimizing training parameters for faster convergence")
        
        if not recommendations:
            recommendations.append("✅ Training completed successfully - no specific recommendations")
        
        return recommendations
    
    def create_comprehensive_report(self) -> Dict[str, Any]:
        """Create comprehensive evaluation report"""
        print("\n📈 Creating comprehensive training evaluation report...")
        
        # Find all training files
        self.find_training_files()
        
        if not self.metrics_files and not self.summary_files:
            return {'error': 'No training files found'}
        
        # Evaluate all sessions
        evaluations = []
        for metrics_file in self.metrics_files:
            evaluation = self.evaluate_training_session(metrics_file)
            if evaluation:
                evaluations.append(evaluation)
        
        # Create comprehensive report
        report = {
            'report_generated': datetime.datetime.now().isoformat(),
            'total_sessions': len(evaluations),
            'individual_evaluations': evaluations,
            'overall_analysis': self.create_overall_analysis(evaluations),
            'comparative_analysis': self.create_comparative_analysis(evaluations),
            'recommendations': self.create_overall_recommendations(evaluations)
        }
        
        # Save report
        report_file = f"training_evaluation_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Comprehensive evaluation report saved to: {report_file}")
        return report
    
    def create_overall_analysis(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create overall analysis from all evaluations"""
        if not evaluations:
            return {}
        
        # Aggregate metrics
        total_sessions = len(evaluations)
        total_images = sum(e.get('ai_images_info', {}).get('total_images', 0) for e in evaluations)
        
        # Average performance
        avg_rewards = []
        avg_detection_losses = []
        avg_relationship_losses = []
        
        for eval_data in evaluations:
            final_results = eval_data.get('final_results', {})
            if final_results:
                avg_rewards.append(final_results.get('reward', 0))
                avg_detection_losses.append(final_results.get('detection_loss', 0))
                avg_relationship_losses.append(final_results.get('relationship_loss', 0))
        
        analysis = {
            'total_sessions': total_sessions,
            'total_ai_images_generated': total_images,
            'average_reward': sum(avg_rewards) / len(avg_rewards) if avg_rewards else 0,
            'average_detection_loss': sum(avg_detection_losses) / len(avg_detection_losses) if avg_detection_losses else 0,
            'average_relationship_loss': sum(avg_relationship_losses) / len(avg_relationship_losses) if avg_relationship_losses else 0,
            'best_session': self.find_best_session(evaluations),
            'worst_session': self.find_worst_session(evaluations)
        }
        
        return analysis
    
    def find_best_session(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find the best performing session"""
        if not evaluations:
            return {}
        
        best_session = None
        best_score = -1
        
        for eval_data in evaluations:
            final_results = eval_data.get('final_results', {})
            reward = final_results.get('reward', 0)
            if reward > best_score:
                best_score = reward
                best_session = eval_data
        
        return best_session
    
    def find_worst_session(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find the worst performing session"""
        if not evaluations:
            return {}
        
        worst_session = None
        worst_score = float('inf')
        
        for eval_data in evaluations:
            final_results = eval_data.get('final_results', {})
            reward = final_results.get('reward', 0)
            if reward < worst_score:
                worst_score = reward
                worst_session = eval_data
        
        return worst_session
    
    def create_comparative_analysis(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create comparative analysis between sessions"""
        if len(evaluations) < 2:
            return {'message': 'Need at least 2 sessions for comparative analysis'}
        
        # Compare performance trends
        rewards = [e.get('final_results', {}).get('reward', 0) for e in evaluations]
        detection_losses = [e.get('final_results', {}).get('detection_loss', 0) for e in evaluations]
        relationship_losses = [e.get('final_results', {}).get('relationship_loss', 0) for e in evaluations]
        
        comparative = {
            'reward_variance': self.calculate_variance(rewards),
            'detection_loss_variance': self.calculate_variance(detection_losses),
            'relationship_loss_variance': self.calculate_variance(relationship_losses),
            'performance_consistency': self.calculate_performance_consistency(evaluations)
        }
        
        return comparative
    
    def calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of values"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance
    
    def calculate_performance_consistency(self, evaluations: List[Dict[str, Any]]) -> float:
        """Calculate performance consistency across sessions"""
        if len(evaluations) < 2:
            return 0.0
        
        rewards = [e.get('final_results', {}).get('reward', 0) for e in evaluations]
        return self.calculate_stability(rewards)
    
    def create_overall_recommendations(self, evaluations: List[Dict[str, Any]]) -> List[str]:
        """Create overall recommendations based on all evaluations"""
        recommendations = []
        
        if not evaluations:
            return ["❌ No training sessions found for analysis"]
        
        # Analyze overall performance
        avg_reward = sum(e.get('final_results', {}).get('reward', 0) for e in evaluations) / len(evaluations)
        
        if avg_reward < 0.3:
            recommendations.append("🔴 Overall low performance - consider fundamental changes to training approach")
        elif avg_reward > 0.7:
            recommendations.append("🟢 Good overall performance - consider fine-tuning for even better results")
        else:
            recommendations.append("🟡 Moderate performance - consider incremental improvements")
        
        # Analyze consistency
        consistency = self.calculate_performance_consistency(evaluations)
        if consistency < 0.5:
            recommendations.append("🔴 Low consistency between sessions - consider standardizing training parameters")
        else:
            recommendations.append("🟢 Good consistency between sessions")
        
        # Analyze AI image generation
        total_images = sum(e.get('ai_images_info', {}).get('total_images', 0) for e in evaluations)
        if total_images < 50:
            recommendations.append("🟡 Low AI image generation - consider increasing image generation for better training")
        
        return recommendations

# Example usage
if __name__ == "__main__":
    evaluator = TrainingEvaluator()
    report = evaluator.create_comprehensive_report()
    print("\n📊 Training Evaluation Complete!")
    print(f"Total sessions evaluated: {report.get('total_sessions', 0)}")
