# File: model_manager.py
import os
import json
import torch
import pickle
from typing import Dict, Any, Optional
import datetime

class ModelManager:
    def __init__(self, base_dir: str = "models"):
        self.base_dir = base_dir
        self.current_experiment_dir = None
        
        # Tạo thư mục models nếu chưa có
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
    
    def set_experiment_dir(self, experiment_dir: str):
        """Set experiment directory for model saving"""
        self.current_experiment_dir = experiment_dir
        models_dir = os.path.join(experiment_dir, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
    
    def save_model_state(self, model_name: str, model_state: Dict[str, Any], epoch: int = None, metadata: Dict[str, Any] = None):
        """Lưu trạng thái model"""
        if not self.current_experiment_dir:
            print("ERROR: No experiment directory set. Call set_experiment_dir() first.")
            return None
        
        models_dir = os.path.join(self.current_experiment_dir, "models")
        
        # Tạo tên file
        if epoch is not None:
            filename = f"{model_name}_epoch_{epoch:02d}.pth"
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{model_name}_{timestamp}.pth"
        
        filepath = os.path.join(models_dir, filename)
        
        # Tạo checkpoint data
        checkpoint = {
            'model_name': model_name,
            'model_state': model_state,
            'epoch': epoch,
            'timestamp': datetime.datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        try:
            # Lưu model state
            torch.save(checkpoint, filepath)
            
            # Lưu metadata riêng
            metadata_file = filepath.replace('.pth', '_metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump({
                    'model_name': model_name,
                    'epoch': epoch,
                    'timestamp': checkpoint['timestamp'],
                    'file_size': os.path.getsize(filepath),
                    **metadata
                }, f, indent=2)
            
            print(f"Saved model state: {filename}")
            return filepath
            
        except Exception as e:
            print(f"ERROR saving model state: {e}")
            return None
    
    def load_model_state(self, model_name: str, experiment_dir: str = None, epoch: int = None):
        """Load trạng thái model"""
        if experiment_dir is None:
            experiment_dir = self.current_experiment_dir
        
        if not experiment_dir or not os.path.exists(experiment_dir):
            print(f"ERROR: Experiment directory not found: {experiment_dir}")
            return None
        
        models_dir = os.path.join(experiment_dir, "models")
        if not os.path.exists(models_dir):
            print(f"ERROR: Models directory not found: {models_dir}")
            return None
        
        # Tìm file model
        model_files = []
        for file in os.listdir(models_dir):
            if file.startswith(f"{model_name}_") and file.endswith('.pth'):
                model_files.append(file)
        
        if not model_files:
            print(f"ERROR: No model files found for {model_name}")
            return None
        
        # Chọn file theo epoch hoặc file mới nhất
        if epoch is not None:
            target_file = f"{model_name}_epoch_{epoch:02d}.pth"
            if target_file in model_files:
                filepath = os.path.join(models_dir, target_file)
            else:
                print(f"ERROR: Model file for epoch {epoch} not found")
                return None
        else:
            # Lấy file mới nhất
            model_files.sort(reverse=True)
            filepath = os.path.join(models_dir, model_files[0])
        
        try:
            # Load model state
            checkpoint = torch.load(filepath, map_location='cpu')
            
            print(f"Loaded model state: {os.path.basename(filepath)}")
            return checkpoint
            
        except Exception as e:
            print(f"ERROR loading model state: {e}")
            return None
    
    def save_training_history(self, history: Dict[str, Any]):
        """Lưu lịch sử training"""
        if not self.current_experiment_dir:
            print("ERROR: No experiment directory set")
            return None
        
        models_dir = os.path.join(self.current_experiment_dir, "models")
        history_file = os.path.join(models_dir, "training_history.json")
        
        try:
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)
            print(f"Saved training history: {history_file}")
            return history_file
        except Exception as e:
            print(f"ERROR saving training history: {e}")
            return None
    
    def get_best_model(self, model_name: str, experiment_dir: str = None, metric: str = 'reward'):
        """Lấy model tốt nhất dựa trên metric"""
        if experiment_dir is None:
            experiment_dir = self.current_experiment_dir
        
        models_dir = os.path.join(experiment_dir, "models")
        if not os.path.exists(models_dir):
            return None
        
        # Load training history
        history_file = os.path.join(models_dir, "training_history.json")
        if not os.path.exists(history_file):
            return None
        
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
            
            # Tìm epoch có metric tốt nhất
            best_epoch = None
            best_value = float('-inf')
            
            for epoch_data in history.get('epochs', []):
                if metric in epoch_data:
                    if epoch_data[metric] > best_value:
                        best_value = epoch_data[metric]
                        best_epoch = epoch_data.get('epoch')
            
            if best_epoch is not None:
                return self.load_model_state(model_name, experiment_dir, best_epoch)
            
        except Exception as e:
            print(f"ERROR getting best model: {e}")
        
        return None
    
    def list_available_models(self, experiment_dir: str = None):
        """Liệt kê các model có sẵn"""
        if experiment_dir is None:
            experiment_dir = self.current_experiment_dir
        
        if not experiment_dir or not os.path.exists(experiment_dir):
            return []
        
        models_dir = os.path.join(experiment_dir, "models")
        if not os.path.exists(models_dir):
            return []
        
        models = []
        for file in os.listdir(models_dir):
            if file.endswith('.pth'):
                filepath = os.path.join(models_dir, file)
                try:
                    # Load metadata
                    metadata_file = filepath.replace('.pth', '_metadata.json')
                    if os.path.exists(metadata_file):
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        models.append({
                            'filename': file,
                            'filepath': filepath,
                            'metadata': metadata
                        })
                except:
                    models.append({
                        'filename': file,
                        'filepath': filepath,
                        'metadata': {}
                    })
        
        return models
    
    def create_model_summary(self, experiment_dir: str = None):
        """Tạo tóm tắt về models trong experiment"""
        if experiment_dir is None:
            experiment_dir = self.current_experiment_dir
        
        models = self.list_available_models(experiment_dir)
        
        summary = {
            'experiment_dir': experiment_dir,
            'total_models': len(models),
            'models': models,
            'created_time': datetime.datetime.now().isoformat()
        }
        
        if experiment_dir:
            summary_file = os.path.join(experiment_dir, "model_summary.json")
            try:
                with open(summary_file, 'w') as f:
                    json.dump(summary, f, indent=2)
                print(f"Created model summary: {summary_file}")
            except Exception as e:
                print(f"ERROR creating model summary: {e}")
        
        return summary
