# File: experiment_manager.py
import os
import json
import shutil
import datetime
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Optional
from PIL import Image
import glob

class ExperimentManager:
    def __init__(self, base_dir: str = "experiments"):
        self.base_dir = base_dir
        self.current_exp_num = self.get_next_experiment_number()
        self.current_exp_dir = None
        
        # Tạo thư mục experiments nếu chưa có
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
    
    def get_next_experiment_number(self) -> int:
        """Lấy số experiment tiếp theo"""
        if not os.path.exists(self.base_dir):
            return 1
        
        # Tìm tất cả thư mục exp_*
        exp_dirs = [d for d in os.listdir(self.base_dir) if d.startswith('exp_')]
        
        if not exp_dirs:
            return 1
        
        # Lấy số cao nhất
        exp_numbers = []
        for exp_dir in exp_dirs:
            try:
                num = int(exp_dir.split('_')[1])
                exp_numbers.append(num)
            except (ValueError, IndexError):
                continue
        
        return max(exp_numbers) + 1 if exp_numbers else 1
    
    def start_new_experiment(self, experiment_name: str = None) -> str:
        """Bắt đầu experiment mới"""
        if experiment_name is None:
            experiment_name = f"exp_{self.current_exp_num:03d}"
        
        self.current_exp_dir = os.path.join(self.base_dir, experiment_name)
        
        # Tạo cấu trúc thư mục
        dirs_to_create = [
            self.current_exp_dir,
            os.path.join(self.current_exp_dir, "ai_images"),
            os.path.join(self.current_exp_dir, "metrics"),
            os.path.join(self.current_exp_dir, "plots"),
            os.path.join(self.current_exp_dir, "models"),
            os.path.join(self.current_exp_dir, "logs")
        ]
        
        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)
        
        # Tạo file metadata
        metadata = {
            "experiment_id": experiment_name,
            "start_time": datetime.datetime.now().isoformat(),
            "status": "running",
            "description": f"Reinforcement Learning Experiment {self.current_exp_num}",
            "version": "1.0"
        }
        
        with open(os.path.join(self.current_exp_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Started new experiment: {experiment_name}")
        print(f"Experiment directory: {self.current_exp_dir}")
        
        return self.current_exp_dir
    
    def save_ai_images(self, ai_images: List[Dict], epoch: int = None):
        """Lưu ảnh AI được tạo ra"""
        if not self.current_exp_dir:
            print("❌ No active experiment. Call start_new_experiment() first.")
            return
        
        ai_images_dir = os.path.join(self.current_exp_dir, "ai_images")
        
        saved_images = []
        for i, img_data in enumerate(ai_images):
            try:
                # Tạo tên file
                if epoch is not None:
                    filename = f"epoch_{epoch:02d}_img_{i:03d}.jpg"
                else:
                    filename = f"img_{i:03d}.jpg"
                
                filepath = os.path.join(ai_images_dir, filename)
                
                # Lưu ảnh
                if isinstance(img_data['image'], Image.Image):
                    img_data['image'].save(filepath, "JPEG", quality=95)
                else:
                    # Nếu là mock image hoặc array
                    if hasattr(img_data['image'], 'save'):
                        img_data['image'].save(filepath, "JPEG", quality=95)
                    else:
                        print(f"⚠️ Cannot save image {i}: unsupported format")
                        continue
                
                # Lưu metadata
                image_metadata = {
                    "filename": filename,
                    "prompt": img_data.get('prompt', ''),
                    "original_relationship": img_data.get('original_relationship', {}),
                    "is_mock": img_data.get('is_mock', False),
                    "epoch": epoch,
                    "saved_time": datetime.datetime.now().isoformat(),
                    "path": filepath
                }

                saved_images.append(image_metadata)

            except Exception as e:
                print(f"❌ Error saving image {i}: {e}")
                continue
        
        # Lưu danh sách ảnh đã lưu
        images_list_file = os.path.join(ai_images_dir, f"images_list_epoch_{epoch:02d}.json" if epoch else "images_list.json")
        with open(images_list_file, 'w') as f:
            json.dump(saved_images, f, indent=2)
        
        print(f"Saved {len(saved_images)} AI images to {ai_images_dir}")
        return saved_images
    
    def save_training_metrics(self, metrics: Dict[str, Any], epoch: int = None):
        """Lưu metrics training"""
        if not self.current_exp_dir:
            print("❌ No active experiment. Call start_new_experiment() first.")
            return
        
        metrics_dir = os.path.join(self.current_exp_dir, "metrics")
        
        # Tạo tên file
        if epoch is not None:
            filename = f"training_metrics_epoch_{epoch:02d}.json"
        else:
            filename = f"training_metrics_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join(metrics_dir, filename)
        
        # Thêm metadata
        enhanced_metrics = {
            "experiment_id": os.path.basename(self.current_exp_dir),
            "saved_time": datetime.datetime.now().isoformat(),
            "epoch": epoch,
            **metrics
        }
        
        with open(filepath, 'w') as f:
            json.dump(enhanced_metrics, f, indent=2)
        
        print(f"Saved training metrics to {filepath}")
        return filepath
    
    def create_training_plots(self, metrics: Dict[str, Any]):
        """Tạo biểu đồ từ metrics"""
        if not self.current_exp_dir:
            print("❌ No active experiment. Call start_new_experiment() first.")
            return
        
        plots_dir = os.path.join(self.current_exp_dir, "plots")
        
        # Lấy dữ liệu training progress
        training_progress = metrics.get('training_progress', [])
        if not training_progress:
            print("⚠️ No training progress data to plot")
            return
        
        # Chuẩn bị dữ liệu
        epochs = [p['epoch'] for p in training_progress]
        detection_losses = [p['detection_loss'] for p in training_progress]
        relationship_losses = [p['relationship_loss'] for p in training_progress]
        rewards = [p['reward'] for p in training_progress]
        ai_images_counts = [p['ai_images_count'] for p in training_progress]
        
        # Tạo figure với subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Results - Experiment {os.path.basename(self.current_exp_dir)}', fontsize=16)
        
        # 1. Loss curves
        axes[0, 0].plot(epochs, detection_losses, 'b-o', label='Detection Loss', linewidth=2, markersize=6)
        axes[0, 0].plot(epochs, relationship_losses, 'r-s', label='Relationship Loss', linewidth=2, markersize=6)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Losses')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Reward curve
        axes[0, 1].plot(epochs, rewards, 'g-^', label='Reward', linewidth=2, markersize=6, color='green')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Reward')
        axes[0, 1].set_title('Training Reward')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. AI Images generated
        axes[1, 0].bar(epochs, ai_images_counts, color='purple', alpha=0.7)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AI Images Generated')
        axes[1, 0].set_title('AI Images Generated per Epoch')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Combined metrics
        ax2 = axes[1, 1]
        ax2_twin = ax2.twinx()
        
        line1 = ax2.plot(epochs, detection_losses, 'b-o', label='Detection Loss', linewidth=2)
        line2 = ax2.plot(epochs, relationship_losses, 'r-s', label='Relationship Loss', linewidth=2)
        line3 = ax2_twin.plot(epochs, rewards, 'g-^', label='Reward', linewidth=2, color='green')
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss', color='blue')
        ax2_twin.set_ylabel('Reward', color='green')
        ax2.set_title('Combined Metrics')
        
        # Combine legends
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Lưu biểu đồ
        plot_file = os.path.join(plots_dir, f"training_plots_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Created training plots: {plot_file}")
        return plot_file
    
    def create_ai_images_grid(self, ai_images: List[Dict], epoch: int = None, max_images: int = 16):
        """Tạo grid ảnh AI"""
        if not self.current_exp_dir:
            print("❌ No active experiment. Call start_new_experiment() first.")
            return
        
        plots_dir = os.path.join(self.current_exp_dir, "plots")
        
        # Lấy ảnh để hiển thị
        display_images = ai_images[:max_images]
        
        if not display_images:
            print("⚠️ No AI images to display")
            return
        
        # Tính toán grid size
        n_images = len(display_images)
        grid_size = int(np.ceil(np.sqrt(n_images)))
        
        # Tạo figure
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
        if grid_size == 1:
            axes = [axes]
        elif grid_size > 1:
            axes = axes.flatten()
        
        fig.suptitle(f'AI Generated Images - Epoch {epoch}' if epoch else 'AI Generated Images', fontsize=16)
        
        for i, img_data in enumerate(display_images):
            if i >= len(axes):
                break
                
            try:
                image_obj = img_data.get('image')
                if isinstance(image_obj, Image.Image):
                    axes[i].imshow(image_obj)
                else:
                    image_path = img_data.get('path') or img_data.get('image_path') or img_data.get('saved_path')
                    if image_path and os.path.exists(image_path):
                        with Image.open(image_path) as loaded_image:
                            axes[i].imshow(loaded_image)
                    else:
                        axes[i].imshow(np.array(img_data.get('image_data', np.zeros((10, 10, 3), dtype=np.uint8))))
                
                # Thêm title với prompt ngắn
                prompt = img_data.get('prompt', '')[:30] + '...' if len(img_data.get('prompt', '')) > 30 else img_data.get('prompt', '')
                axes[i].set_title(prompt, fontsize=8)
                axes[i].axis('off')
                
            except Exception as e:
                print(f"⚠️ Error displaying image {i}: {e}")
                axes[i].text(0.5, 0.5, f'Error\n{str(e)[:20]}...', 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].axis('off')
        
        # Ẩn các axes không sử dụng
        for i in range(len(display_images), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        # Lưu grid
        grid_file = os.path.join(plots_dir, f"ai_images_grid_epoch_{epoch:02d}.png" if epoch else "ai_images_grid.png")
        plt.savefig(grid_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Created AI images grid: {grid_file}")
        return grid_file
    
    def finalize_experiment(self, final_results: Dict[str, Any] = None):
        """Hoàn thành experiment"""
        if not self.current_exp_dir:
            print("❌ No active experiment to finalize.")
            return
        
        # Cập nhật metadata
        metadata_file = os.path.join(self.current_exp_dir, "metadata.json")
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        metadata.update({
            "end_time": datetime.datetime.now().isoformat(),
            "status": "completed",
            "final_results": final_results or {}
        })
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Tạo summary report
        self.create_experiment_summary()
        
        print(f"Experiment completed: {os.path.basename(self.current_exp_dir)}")
        print(f"All results saved to: {self.current_exp_dir}")
        
        # Reset current experiment
        self.current_exp_dir = None
        self.current_exp_num += 1
    
    def create_experiment_summary(self):
        """Tạo báo cáo tổng kết experiment"""
        if not self.current_exp_dir:
            return
        
        summary_file = os.path.join(self.current_exp_dir, "experiment_summary.json")
        
        # Thu thập thông tin
        summary = {
            "experiment_id": os.path.basename(self.current_exp_dir),
            "created_time": datetime.datetime.now().isoformat(),
            "directories": {
                "ai_images": len(glob.glob(os.path.join(self.current_exp_dir, "ai_images", "*.jpg"))),
                "metrics_files": len(glob.glob(os.path.join(self.current_exp_dir, "metrics", "*.json"))),
                "plot_files": len(glob.glob(os.path.join(self.current_exp_dir, "plots", "*.png")))
            },
            "files_created": self.get_experiment_files()
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def get_experiment_files(self) -> List[str]:
        """Lấy danh sách files trong experiment"""
        if not self.current_exp_dir:
            return []
        
        files = []
        for root, dirs, filenames in os.walk(self.current_exp_dir):
            for filename in filenames:
                rel_path = os.path.relpath(os.path.join(root, filename), self.current_exp_dir)
                files.append(rel_path)
        
        return files
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """Liệt kê tất cả experiments"""
        if not os.path.exists(self.base_dir):
            return []
        
        experiments = []
        for exp_dir in os.listdir(self.base_dir):
            exp_path = os.path.join(self.base_dir, exp_dir)
            if os.path.isdir(exp_path):
                metadata_file = os.path.join(exp_path, "metadata.json")
                if os.path.exists(metadata_file):
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        experiments.append({
                            "experiment_id": exp_dir,
                            "path": exp_path,
                            **metadata
                        })
                    except:
                        continue
        
        return sorted(experiments, key=lambda x: x.get('start_time', ''), reverse=True)
    
    def load_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Load thông tin experiment"""
        exp_path = os.path.join(self.base_dir, experiment_id)
        if not os.path.exists(exp_path):
            return {}
        
        # Load metadata
        metadata_file = os.path.join(exp_path, "metadata.json")
        metadata = {}
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        
        # Load metrics files
        metrics_files = glob.glob(os.path.join(exp_path, "metrics", "*.json"))
        metrics_data = []
        for metrics_file in metrics_files:
            try:
                with open(metrics_file, 'r') as f:
                    metrics_data.append(json.load(f))
            except:
                continue
        
        # Load AI images info
        ai_images_dir = os.path.join(exp_path, "ai_images")
        ai_images_info = []
        if os.path.exists(ai_images_dir):
            for img_file in glob.glob(os.path.join(ai_images_dir, "*.jpg")):
                ai_images_info.append({
                    "filename": os.path.basename(img_file),
                    "path": img_file
                })
        
        return {
            "experiment_id": experiment_id,
            "path": exp_path,
            "metadata": metadata,
            "metrics": metrics_data,
            "ai_images": ai_images_info,
            "plots": glob.glob(os.path.join(exp_path, "plots", "*.png"))
        }
