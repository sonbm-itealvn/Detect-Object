# File: experiment_viewer.py
import os
import json
import glob
from typing import Dict, List, Any
from RL.experiment_manager import ExperimentManager
import matplotlib.pyplot as plt
from PIL import Image

class ExperimentViewer:
    def __init__(self, base_dir: str = "experiments"):
        self.experiment_manager = ExperimentManager(base_dir)
        self.base_dir = base_dir
    
    def list_all_experiments(self) -> List[Dict[str, Any]]:
        """Liệt kê tất cả experiments"""
        experiments = self.experiment_manager.list_experiments()
        
        print("DANH SACH EXPERIMENTS")
        print("=" * 80)
        
        if not experiments:
            print("Khong co experiment nao duoc tim thay")
            return []
        
        for i, exp in enumerate(experiments, 1):
            print(f"\nExperiment {i}: {exp['experiment_id']}")
            print(f"   Thoi gian: {exp.get('start_time', 'Unknown')}")
            print(f"   Trang thai: {exp.get('status', 'Unknown')}")
            print(f"   Duong dan: {exp.get('path', 'Unknown')}")
            
            # Hiển thị thông tin chi tiết nếu có
            if 'final_results' in exp:
                results = exp['final_results']
                print(f"   Final Reward: {results.get('reward', 0):.4f}")
                print(f"   Detection Loss: {results.get('detection_loss', 0):.4f}")
                print(f"   Relationship Loss: {results.get('relationship_loss', 0):.4f}")
        
        return experiments
    
    def view_experiment_details(self, experiment_id: str):
        """Xem chi tiết một experiment"""
        exp_data = self.experiment_manager.load_experiment(experiment_id)
        
        if not exp_data:
            print(f"❌ Không tìm thấy experiment: {experiment_id}")
            return
        
        print(f"\n🔬 CHI TIẾT EXPERIMENT: {experiment_id}")
        print("=" * 60)
        
        # Metadata
        metadata = exp_data.get('metadata', {})
        print(f"\n📋 THÔNG TIN CƠ BẢN:")
        print(f"   🆔 ID: {metadata.get('experiment_id', 'Unknown')}")
        print(f"   📅 Bắt đầu: {metadata.get('start_time', 'Unknown')}")
        print(f"   📅 Kết thúc: {metadata.get('end_time', 'Unknown')}")
        print(f"   📊 Trạng thái: {metadata.get('status', 'Unknown')}")
        print(f"   📝 Mô tả: {metadata.get('description', 'Unknown')}")
        
        # Final results
        if 'final_results' in metadata:
            results = metadata['final_results']
            print(f"\n🎯 KẾT QUẢ CUỐI CÙNG:")
            print(f"   🏆 Reward: {results.get('reward', 0):.4f}")
            print(f"   📈 Detection Loss: {results.get('detection_loss', 0):.4f}")
            print(f"   🔗 Relationship Loss: {results.get('relationship_loss', 0):.4f}")
            print(f"   🎨 AI Images: {results.get('total_ai_images', 0)}")
            print(f"   ⏱️ Thời gian: {results.get('training_duration', 0):.2f}s")
        
        # Metrics files
        metrics = exp_data.get('metrics', [])
        print(f"\n📊 METRICS FILES: {len(metrics)} files")
        for i, metric in enumerate(metrics, 1):
            print(f"   {i}. Epoch {metric.get('epoch', 'Unknown')} - {metric.get('timestamp', 'Unknown')}")
        
        # AI Images
        ai_images = exp_data.get('ai_images', [])
        print(f"\n🖼️ AI IMAGES: {len(ai_images)} images")
        
        # Plots
        plots = exp_data.get('plots', [])
        print(f"\n📈 PLOTS: {len(plots)} files")
        for plot in plots:
            print(f"   📊 {os.path.basename(plot)}")
    
    def show_experiment_plots(self, experiment_id: str):
        """Hiển thị biểu đồ của experiment"""
        exp_data = self.experiment_manager.load_experiment(experiment_id)
        
        if not exp_data:
            print(f"❌ Không tìm thấy experiment: {experiment_id}")
            return
        
        plots = exp_data.get('plots', [])
        if not plots:
            print("❌ Không có biểu đồ nào trong experiment này")
            return
        
        print(f"\n📈 BIỂU ĐỒ EXPERIMENT: {experiment_id}")
        print("=" * 50)
        
        for i, plot_path in enumerate(plots, 1):
            print(f"{i}. {os.path.basename(plot_path)}")
            print(f"   📁 Đường dẫn: {plot_path}")
        
        # Hỏi user muốn xem biểu đồ nào
        try:
            choice = input(f"\nChọn biểu đồ để xem (1-{len(plots)}, hoặc Enter để bỏ qua): ").strip()
            if choice and choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(plots):
                    self.display_plot(plots[idx])
        except KeyboardInterrupt:
            print("\n👋 Đã hủy")
    
    def display_plot(self, plot_path: str):
        """Hiển thị một biểu đồ"""
        try:
            if not os.path.exists(plot_path):
                print(f"❌ File không tồn tại: {plot_path}")
                return
            
            # Load và hiển thị ảnh
            img = Image.open(plot_path)
            img.show()
            print(f"✅ Đã mở biểu đồ: {os.path.basename(plot_path)}")
            
        except Exception as e:
            print(f"❌ Lỗi hiển thị biểu đồ: {e}")
    
    def show_ai_images(self, experiment_id: str, max_images: int = 9):
        """Hiển thị ảnh AI của experiment"""
        exp_data = self.experiment_manager.load_experiment(experiment_id)
        
        if not exp_data:
            print(f"❌ Không tìm thấy experiment: {experiment_id}")
            return
        
        ai_images = exp_data.get('ai_images', [])
        if not ai_images:
            print("❌ Không có ảnh AI nào trong experiment này")
            return
        
        print(f"\n🖼️ ẢNH AI EXPERIMENT: {experiment_id}")
        print("=" * 50)
        print(f"📊 Tổng số ảnh: {len(ai_images)}")
        
        # Hiển thị một số ảnh đầu tiên
        display_images = ai_images[:max_images]
        
        for i, img_info in enumerate(display_images, 1):
            print(f"\n{i}. {img_info['filename']}")
            print(f"   📁 Đường dẫn: {img_info['path']}")
            
            # Hỏi user có muốn xem ảnh không
            try:
                view = input(f"   Xem ảnh {i}? (y/n, hoặc Enter để bỏ qua): ").strip().lower()
                if view == 'y':
                    self.display_image(img_info['path'])
            except KeyboardInterrupt:
                print("\n👋 Đã hủy")
                break
    
    def display_image(self, image_path: str):
        """Hiển thị một ảnh"""
        try:
            if not os.path.exists(image_path):
                print(f"❌ File không tồn tại: {image_path}")
                return
            
            # Load và hiển thị ảnh
            img = Image.open(image_path)
            img.show()
            print(f"✅ Đã mở ảnh: {os.path.basename(image_path)}")
            
        except Exception as e:
            print(f"❌ Lỗi hiển thị ảnh: {e}")
    
    def compare_experiments(self, experiment_ids: List[str]):
        """So sánh nhiều experiments"""
        if len(experiment_ids) < 2:
            print("❌ Cần ít nhất 2 experiments để so sánh")
            return
        
        print(f"\n📊 SO SÁNH EXPERIMENTS")
        print("=" * 60)
        
        experiments_data = []
        for exp_id in experiment_ids:
            exp_data = self.experiment_manager.load_experiment(exp_id)
            if exp_data:
                experiments_data.append(exp_data)
            else:
                print(f"⚠️ Không tìm thấy experiment: {exp_id}")
        
        if len(experiments_data) < 2:
            print("❌ Không đủ experiments để so sánh")
            return
        
        # So sánh metrics
        print(f"\n📈 BẢNG SO SÁNH:")
        print("-" * 80)
        print(f"{'Metric':<25} {'Experiment 1':<15} {'Experiment 2':<15} {'Experiment 3':<15}")
        print("-" * 80)
        
        for i, exp_data in enumerate(experiments_data, 1):
            metadata = exp_data.get('metadata', {})
            final_results = metadata.get('final_results', {})
            
            print(f"Experiment {i}: {exp_data['experiment_id']}")
            print(f"  Reward: {final_results.get('reward', 0):.4f}")
            print(f"  Detection Loss: {final_results.get('detection_loss', 0):.4f}")
            print(f"  Relationship Loss: {final_results.get('relationship_loss', 0):.4f}")
            print(f"  AI Images: {final_results.get('total_ai_images', 0)}")
            print(f"  Duration: {final_results.get('training_duration', 0):.2f}s")
            print()
    
    def export_experiment(self, experiment_id: str, export_dir: str = None):
        """Export experiment ra thư mục khác"""
        exp_data = self.experiment_manager.load_experiment(experiment_id)
        
        if not exp_data:
            print(f"❌ Không tìm thấy experiment: {experiment_id}")
            return
        
        if export_dir is None:
            export_dir = f"exported_{experiment_id}"
        
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        
        exp_path = exp_data['path']
        
        # Copy tất cả files
        import shutil
        for root, dirs, files in os.walk(exp_path):
            for file in files:
                src_path = os.path.join(root, file)
                rel_path = os.path.relpath(src_path, exp_path)
                dst_path = os.path.join(export_dir, rel_path)
                
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.copy2(src_path, dst_path)
        
        print(f"✅ Đã export experiment {experiment_id} ra {export_dir}")
    
    def interactive_menu(self):
        """Menu tương tác"""
        while True:
            print("\n🔬 EXPERIMENT VIEWER")
            print("=" * 40)
            print("1. Liệt kê tất cả experiments")
            print("2. Xem chi tiết experiment")
            print("3. Xem biểu đồ experiment")
            print("4. Xem ảnh AI experiment")
            print("5. So sánh experiments")
            print("6. Export experiment")
            print("7. Thoát")
            
            try:
                choice = input("\nChọn chức năng (1-7): ").strip()
                
                if choice == "1":
                    self.list_all_experiments()
                
                elif choice == "2":
                    exp_id = input("Nhập experiment ID: ").strip()
                    self.view_experiment_details(exp_id)
                
                elif choice == "3":
                    exp_id = input("Nhập experiment ID: ").strip()
                    self.show_experiment_plots(exp_id)
                
                elif choice == "4":
                    exp_id = input("Nhập experiment ID: ").strip()
                    self.show_ai_images(exp_id)
                
                elif choice == "5":
                    exp_ids = input("Nhập experiment IDs (cách nhau bởi dấu phẩy): ").strip()
                    exp_id_list = [id.strip() for id in exp_ids.split(',')]
                    self.compare_experiments(exp_id_list)
                
                elif choice == "6":
                    exp_id = input("Nhập experiment ID: ").strip()
                    export_dir = input("Nhập thư mục export (Enter để dùng mặc định): ").strip()
                    if not export_dir:
                        export_dir = None
                    self.export_experiment(exp_id, export_dir)
                
                elif choice == "7":
                    print("👋 Tạm biệt!")
                    break
                
                else:
                    print("❌ Lựa chọn không hợp lệ!")
            
            except KeyboardInterrupt:
                print("\n👋 Tạm biệt!")
                break
            except Exception as e:
                print(f"❌ Lỗi: {e}")

# Example usage
if __name__ == "__main__":
    viewer = ExperimentViewer()
    viewer.interactive_menu()
