import json
import subprocess
import threading
import re
import os
import glob
from sentence_transformers import SentenceTransformer, util
from PIL import Image, ImageDraw

from RL.rl_enhancement import AppReinforcementLearning
from RL.training_evaluator import TrainingEvaluator
from RL.experiment_viewer import ExperimentViewer

class ObjectDetectionConsoleApp:
    def __init__(self):
        # Các đường dẫn mặc định
        self.image_path = None
        self.result_image_path = "result.jpg"
        self.result_json_path = "converted_bboxes.json"
        self.relationship_json_path = "relationships.json"
        self.checkpoint_path = "checkpoint.pth"  # Sử dụng file checkpoint mặc định

        # Load model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Initialize RL components
        self.rl_enhancement = AppReinforcementLearning(self)
        self.training_evaluator = TrainingEvaluator()
        self.experiment_viewer = ExperimentViewer()

    def set_image_path(self, image_path):
        """Set the image path for processing"""
        if os.path.exists(image_path):
            self.image_path = image_path
            print(f"✅ Đã chọn ảnh: {image_path}")
            return True
        else:
            print(f"❌ File không tồn tại: {image_path}")
            return False

    def load_and_display_objects(self):
        """Tải và hiển thị danh sách vật thể từ JSON"""
        try:
            with open(self.result_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Kiểm tra format của data
            if isinstance(data, list) and len(data) > 0:
                # Nếu là list, lấy phần tử đầu tiên
                if "objects" in data[0]:
                    objects = data[0]["objects"]
                else:
                    # Nếu không có key "objects", có thể data là list các object trực tiếp
                    objects = data
            elif isinstance(data, dict):
                objects = data.get("objects", [])
            else:
                objects = []

            if not objects:
                print("❌ Không có vật thể nào được phát hiện")
                return

            print("\n📦 VẬT THỂ ĐƯỢC PHÁT HIỆN:")
            print("=" * 50)
            
            # Hiển thị thông tin vật thể
            for i, obj in enumerate(objects, 1):
                class_name = obj.get("class", "Unknown")
                bbox = obj.get("bbox", [])
                
                if len(bbox) >= 4:
                    x, y, w, h = bbox[:4]
                    print(f"🔸 {i}. {class_name.upper()}")
                    print(f"   📍 Vị trí: ({x}, {y})")
                    print(f"   📏 Kích thước: {w-x} x {h-y}")
                    print(f"   🎯 Độ tin cậy: {obj.get('confidence', 'N/A')}")
                    print()
                else:
                    print(f"🔸 {i}. {class_name.upper()}")
                    print(f"   📍 Thông tin bbox không hợp lệ")
                    print()
                
        except FileNotFoundError:
            print(f"❌ Không tìm thấy file: {self.result_json_path}")
        except json.JSONDecodeError:
            print("❌ Lỗi đọc file JSON")
        except Exception as e:
            print(f"❌ Lỗi: {str(e)}")

    def load_and_display_relationships(self):
        """Tải và hiển thị danh sách mối quan hệ từ JSON (đã được lọc sẵn)"""
        try:
            with open(self.relationship_json_path, "r", encoding="utf-8") as f:
                relationships = json.load(f)
            
            if not relationships:
                print("❌ Không có mối quan hệ nào được phát hiện")
                return

            print(f"\n🔗 MỐI QUAN HỆ (Đã lọc ngưỡng 0.5):")
            print("=" * 50)
            print(f"📊 Hiển thị {len(relationships)} mối quan hệ")
            print()
            
            # Hiển thị thông tin mối quan hệ (đã được lọc sẵn)
            for i, rel in enumerate(relationships, 1):
                subject = rel.get("subject", "Unknown")
                relation = rel.get("relation", "Unknown")
                obj = rel.get("object", "Unknown")
                similarity = rel.get("visual_similarity", 0)
                
                # Màu sắc dựa trên độ tin cậy
                if similarity > 0:
                    confidence_color = "🟢" if similarity > 0.7 else "🟡" if similarity > 0.4 else "🔴"
                    print(f"{confidence_color} {i}. {subject.upper()}")
                    print(f"   🔗 {relation.upper()}")
                    print(f"   🎯 {obj.upper()}")
                    print(f"   📊 Độ tin cậy: {similarity:.2f}")
                    print()
                else:
                    # Nếu không có visual_similarity, hiển thị đơn giản
                    print(f"🔸 {i}. {subject.upper()}")
                    print(f"   🔗 {relation.upper()}")
                    print(f"   🎯 {obj.upper()}")
                    print()
                
        except FileNotFoundError:
            print(f"❌ Không tìm thấy file: {self.relationship_json_path}")
        except json.JSONDecodeError:
            print("❌ Lỗi đọc file JSON")
        except Exception as e:
            print(f"❌ Lỗi: {str(e)}")

    def refresh_data(self):
        """Tải lại dữ liệu JSON mà không cần chạy lại pipeline"""
        print("🔄 Đang tải lại dữ liệu...")
        self.load_and_display_objects()
        self.load_and_display_relationships()
        print("✅ Đã tải lại dữ liệu thành công!")

    def draw_relationship_boxes_on_image(self):
        """Vẽ bbox mối quan hệ trên ảnh kết quả"""
        try:
            # Đọc dữ liệu vật thể
            with open(self.result_json_path, "r", encoding="utf-8") as f:
                objects_data = json.load(f)
            
            # Đọc dữ liệu mối quan hệ
            with open(self.relationship_json_path, "r", encoding="utf-8") as f:
                relationships_data = json.load(f)
            
            # Lấy danh sách objects
            if isinstance(objects_data, list) and len(objects_data) > 0:
                if "objects" in objects_data[0]:
                    objects = objects_data[0]["objects"]
                else:
                    objects = objects_data
            elif isinstance(objects_data, dict):
                objects = objects_data.get("objects", [])
            else:
                objects = []
            
            if not objects or not relationships_data:
                print("❌ Không có dữ liệu để vẽ bbox")
                return
            
            # Tìm ảnh kết quả
            image_dir = os.path.dirname(self.image_path)
            image_id = os.path.splitext(os.path.basename(self.image_path))[0]
            output_images = glob.glob(f"**/output_{image_id}.jpg", recursive=True)
            
            if not output_images:
                print("❌ Không tìm thấy ảnh kết quả để vẽ bbox")
                return
            
            latest_result = max(output_images, key=os.path.getmtime)
            
            # Mở ảnh và vẽ bbox
            image = Image.open(latest_result)
            draw = ImageDraw.Draw(image)
            
            # Tạo dictionary để tìm object theo tên
            def _register_object(bucket_map, obj_entry):
                class_name = obj_entry.get("class", "").lower()
                if not class_name:
                    return
                bucket = bucket_map.setdefault(class_name, {"items": [], "cursor": 0})
                bucket["items"].append(obj_entry)

            def _next_object(bucket):
                if not bucket or not bucket["items"]:
                    return None
                idx = bucket["cursor"] % len(bucket["items"])
                bucket["cursor"] = (bucket["cursor"] + 1) % len(bucket["items"])
                return bucket["items"][idx]

            objects_dict = {}
            for obj in objects:
                _register_object(objects_dict, obj)
            
            # Vẽ bbox cho từng mối quan hệ
            colors = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "cyan"]
            
            for i, rel in enumerate(relationships_data):
                subject = rel.get("subject", "").lower()
                obj = rel.get("object", "").lower()
                
                color = colors[i % len(colors)]
                
                # Vẽ bbox cho subject
                subject_entry = _next_object(objects_dict.get(subject))
                if subject_entry:
                    bbox = subject_entry.get("bbox", [])
                    if len(bbox) >= 4:
                        x, y, w, h = bbox[:4]
                        # Vẽ box đậm hơn với width=6
                        draw.rectangle([x, y, w, h], outline=color, width=6)
                        # Vẽ chữ to hơn với font size 24
                        try:
                            from PIL import ImageFont
                            # Thử sử dụng font mặc định với size lớn hơn
                            font = ImageFont.truetype("arial.ttf", 24)
                        except:
                            # Nếu không tìm thấy font, sử dụng font mặc định
                            font = ImageFont.load_default()
                        draw.text((x, y - 30), f"S: {subject.upper()}", fill=color, font=font)
                
                # Vẽ bbox cho object
                object_entry = _next_object(objects_dict.get(obj))
                if object_entry:
                    bbox = object_entry.get("bbox", [])
                    if len(bbox) >= 4:
                        x, y, w, h = bbox[:4]
                        # Vẽ box đậm hơn với width=6
                        draw.rectangle([x, y, w, h], outline=color, width=6)
                        # Vẽ chữ to hơn với font size 24
                        try:
                            from PIL import ImageFont
                            # Thử sử dụng font mặc định với size lớn hơn
                            font = ImageFont.truetype("arial.ttf", 24)
                        except:
                            # Nếu không tìm thấy font, sử dụng font mặc định
                            font = ImageFont.load_default()
                        draw.text((x, y - 30), f"O: {obj.upper()}", fill=color, font=font)
            
            # Lưu ảnh với bbox
            result_path = f"relationship_result_{image_id}.jpg"
            image.save(result_path)
            
            print(f"✅ Đã vẽ bbox mối quan hệ và lưu tại: {result_path}")
            
        except Exception as e:
            print(f"❌ Lỗi khi vẽ bbox mối quan hệ: {e}")

    def run_pipeline(self):
        """Chạy toàn bộ pipeline xử lý ảnh"""
        if not self.image_path:
            print("❌ Hãy chọn ảnh trước!")
            return

        print("⏳ Đang xử lý... Vui lòng chờ.")

        try:
            # 1️⃣ Chạy detect_objects.py
            print("🔍 Đang phát hiện vật thể...")
            subprocess.run(["python", "detect_objects.py", self.image_path], check=True)

            # 2️⃣ Chạy convert_yolo_to_reltr.py (sau khi detect_objects.py hoàn tất)
            print("🔄 Đang chuyển đổi dữ liệu YOLO...")
            subprocess.run(["python", "convert_yolo_to_reltr.py", "result.json"], check=True)

            # 3️⃣ Chạy boundingbox_objects.py (sau khi convert_yolo_to_reltr.py hoàn tất)
            print("🔗 Đang xác định mối quan hệ giữa các vật thể...")
            subprocess.run(["python", "boundingbox_objects.py", "--yolo_json", self.result_json_path,"--img_path",self.image_path,"--device","cpu", "--resume", self.checkpoint_path], check=True)

            image_dir = os.path.dirname(self.image_path)
            image_id = os.path.splitext(os.path.basename(self.image_path))[0]

            # ✅ Tìm ảnh output_anh2.jpg ở bất kỳ thư mục nào
            output_images = glob.glob(f"**/output_{image_id}.jpg", recursive=True)

            if output_images:
                latest_result = max(output_images, key=os.path.getmtime)  # Lấy ảnh mới nhất nếu có nhiều ảnh trùng tên
                print(f"✅ Tìm thấy ảnh kết quả: {latest_result}")
            else:
                print("📂 Danh sách file trong thư mục:", os.listdir(image_dir))  # Debug kiểm tra
                print("❌ Không tìm thấy ảnh kết quả!")

            # 4️⃣ Tải và hiển thị dữ liệu JSON
            print("📊 Đang tải dữ liệu kết quả...")
            self.load_and_display_objects()
            self.load_and_display_relationships()
            
            # 5️⃣ Vẽ bbox mối quan hệ trên ảnh
            print("🎨 Đang vẽ bbox mối quan hệ...")
            self.draw_relationship_boxes_on_image()
            print("✅ Hoàn tất! Dữ liệu đã được tải và vẽ bbox.")
            
        except Exception as e:
            print(f"❌ Lỗi: {e}")

    def run_rl_training(self, epochs=5):
        """Run reinforcement learning training"""
        print(f"\n🧠 RL TRAINING")
        print("=" * 60)
        print(f"Epochs: {epochs}")
        print()
        
        # Menu chọn dataset
        print("📂 CHỌN DATASET ĐỂ TRAINING:")
        print("1. Sử dụng dataset hiện tại (nếu đã có)")
        print("2. Chọn thư mục chứa ảnh để build dataset")
        print("3. Bỏ qua (sẽ dùng dataset từ relationships hiện tại)")
        print()
        
        dataset_choice = input("Chọn phương thức (1-3, mặc định 3): ").strip() or "3"
        
        dataset_dir = None
        if dataset_choice == "2":
            dataset_dir = input("Nhập đường dẫn thư mục chứa ảnh: ").strip()
            if not dataset_dir or not os.path.exists(dataset_dir):
                print("❌ Thư mục không tồn tại! Sẽ bỏ qua dataset directory.")
                dataset_dir = None
            else:
                print(f"✅ Đã chọn thư mục dataset: {dataset_dir}")
                # Build dataset từ thư mục ảnh
                try:
                    print("📦 Đang build dataset từ thư mục ảnh...")
                    if not self.rl_enhancement.rl_system:
                        self.rl_enhancement.setup_reinforcement_learning()
                    rl_agent = self.rl_enhancement.rl_system
                    if rl_agent:
                        count = rl_agent.build_dataset_from_directory(dataset_dir)
                        print(f"✅ Đã build dataset với {count} samples!")
                    else:
                        print("⚠️  RL agent chưa được khởi tạo")
                except Exception as e:
                    print(f"⚠️  Lỗi build dataset: {e}")
                    print("   Sẽ tiếp tục với dataset hiện tại...")
        
        print(f"\n🚀 Bắt đầu RL Training...")
        try:
            results = self.rl_enhancement.run_reinforcement_learning(epochs=epochs, image_directory=dataset_dir)
            print(f"\n✅ RL Training hoàn tất!")
            print(f"🎯 Final Reward: {results.get('reward', 0):.4f}")
            print(f"📈 Total AI Images: {results.get('total_ai_images', 0)}")
            if hasattr(self.rl_enhancement, 'experience_manager'):
                buffer_stats = self.rl_enhancement.experience_manager.stats()
                print(f"💾 Replay buffer size: {buffer_stats['buffer_size']} / {buffer_stats['capacity']}")
        except Exception as e:
            print(f"❌ RL Training lỗi: {e}")
            import traceback
            traceback.print_exc()
    
    def generate_synthetic_data(self):
        """Generate synthetic dataset from relationships"""
        print("🎨 Đang tạo dữ liệu synthetic...")
        
        try:
            synthetic_data = self.rl_enhancement.generate_synthetic_dataset()
            print(f"✅ Đã tạo {len(synthetic_data)} ảnh synthetic!")
        except Exception as e:
            print(f"❌ Tạo synthetic data lỗi: {e}")
    
    def evaluate_training_results(self):
        """Evaluate training results and create comprehensive report"""
        print("📊 Đang đánh giá kết quả training...")
        
        try:
            # Create comprehensive evaluation report
            report = self.training_evaluator.create_comprehensive_report()
            
            if 'error' in report:
                print(f"❌ {report['error']}")
                return
            
            # Display evaluation results
            total_sessions = report.get('total_sessions', 0)
            overall_analysis = report.get('overall_analysis', {})
            
            if total_sessions > 0:
                avg_reward = overall_analysis.get('average_reward', 0)
                total_images = overall_analysis.get('total_ai_images_generated', 0)
                
                print(f"✅ Đánh giá hoàn tất! {total_sessions} sessions, Reward: {avg_reward:.3f}, AI Images: {total_images}")
                
                # Display detailed results in console
                print("\n" + "="*60)
                print("📊 TRAINING EVALUATION RESULTS")
                print("="*60)
                print(f"Total Training Sessions: {total_sessions}")
                print(f"Total AI Images Generated: {total_images}")
                print(f"Average Reward Score: {avg_reward:.4f}")
                print(f"Average Detection Loss: {overall_analysis.get('average_detection_loss', 0):.4f}")
                print(f"Average Relationship Loss: {overall_analysis.get('average_relationship_loss', 0):.4f}")
                
                # Show recommendations
                recommendations = report.get('recommendations', [])
                if recommendations:
                    print("\n📋 RECOMMENDATIONS:")
                    for rec in recommendations:
                        print(f"  {rec}")
                
                print("\n📁 Files created:")
                print("  - rl_training_metrics_*.json (detailed metrics)")
                print("  - rl_training_summary_*.json (training summaries)")
                print("  - training_evaluation_report_*.json (comprehensive report)")
                print("="*60)
                
            else:
                print("❌ Không tìm thấy kết quả training để đánh giá")
                
        except Exception as e:
            print(f"❌ Lỗi đánh giá: {e}")
    
    def list_experiments(self):
        """Liệt kê tất cả experiments"""
        print("\n🔬 QUẢN LÝ EXPERIMENTS")
        print("=" * 50)
        self.experiment_viewer.list_all_experiments()
    
    def view_experiment(self, experiment_id: str = None):
        """Xem chi tiết experiment"""
        if experiment_id is None:
            experiment_id = input("Nhập experiment ID: ").strip()
        
        if not experiment_id:
            print("❌ Cần nhập experiment ID")
            return
        
        print(f"\n🔬 XEM CHI TIẾT EXPERIMENT: {experiment_id}")
        print("=" * 50)
        self.experiment_viewer.view_experiment_details(experiment_id)
    
    def show_experiment_plots(self, experiment_id: str = None):
        """Xem biểu đồ experiment"""
        if experiment_id is None:
            experiment_id = input("Nhập experiment ID: ").strip()
        
        if not experiment_id:
            print("❌ Cần nhập experiment ID")
            return
        
        print(f"\n📈 BIỂU ĐỒ EXPERIMENT: {experiment_id}")
        print("=" * 50)
        self.experiment_viewer.show_experiment_plots(experiment_id)
    
    def show_experiment_images(self, experiment_id: str = None):
        """Xem ảnh AI của experiment"""
        if experiment_id is None:
            experiment_id = input("Nhập experiment ID: ").strip()
        
        if not experiment_id:
            print("❌ Cần nhập experiment ID")
            return
        
        print(f"\n🖼️ ẢNH AI EXPERIMENT: {experiment_id}")
        print("=" * 50)
        self.experiment_viewer.show_ai_images(experiment_id)
    
    def compare_experiments(self):
        """So sánh experiments"""
        print("\n📊 SO SÁNH EXPERIMENTS")
        print("=" * 50)
        
        exp_ids = input("Nhập experiment IDs (cách nhau bởi dấu phẩy): ").strip()
        if not exp_ids:
            print("❌ Cần nhập experiment IDs")
            return
        
        exp_id_list = [id.strip() for id in exp_ids.split(',')]
        self.experiment_viewer.compare_experiments(exp_id_list)
    
    def export_experiment(self, experiment_id: str = None):
        """Export experiment"""
        if experiment_id is None:
            experiment_id = input("Nhập experiment ID: ").strip()
        
        if not experiment_id:
            print("❌ Cần nhập experiment ID")
            return
        
        export_dir = input("Nhập thư mục export (Enter để dùng mặc định): ").strip()
        if not export_dir:
            export_dir = None
        
        print(f"\n📤 EXPORT EXPERIMENT: {experiment_id}")
        print("=" * 50)
        self.experiment_viewer.export_experiment(experiment_id, export_dir)

def main():
    """Main function to run the console application"""
    app = ObjectDetectionConsoleApp()
    
    print("🔍 Object Detection & Relationship Analysis - Console Version")
    print("=" * 60)
    
    while True:
        print("\n📋 MENU:")
        print("1. Chọn ảnh để xử lý")
        print("2. Chạy pipeline phát hiện vật thể và mối quan hệ")
        print("3. Tải lại dữ liệu JSON")
        print("4. Chạy RL Training")
        print("5. Tạo dữ liệu synthetic")
        print("6. Đánh giá kết quả training")
        print("7. Quản lý Experiments")
        print("8. Tiếp tục RL Training từ experiment trước")
        print("9. Thoát")
        
        choice = input("\nNhập lựa chọn (1-9): ").strip()
        
        if choice == "1":
            image_path = input("Nhập đường dẫn ảnh: ").strip()
            app.set_image_path(image_path)
            
        elif choice == "2":
            if not app.image_path:
                print("❌ Hãy chọn ảnh trước!")
                continue
            app.run_pipeline()
            
        elif choice == "3":
            app.refresh_data()
            
        elif choice == "4":
            print("\n🧠 RL TRAINING")
            print("=" * 30)
            try:
                epochs = int(input("Nhập số epochs (mặc định 5): ") or "5")
                if epochs <= 0:
                    print("❌ Số epochs phải lớn hơn 0!")
                    continue
                app.run_rl_training(epochs)
            except ValueError:
                print("❌ Vui lòng nhập số hợp lệ!")
                continue
            
        elif choice == "5":
            app.generate_synthetic_data()
            
        elif choice == "6":
            app.evaluate_training_results()
            
        elif choice == "7":
            # Quản lý Experiments
            while True:
                print("\n🔬 QUẢN LÝ EXPERIMENTS")
                print("=" * 40)
                print("1. Liệt kê tất cả experiments")
                print("2. Xem chi tiết experiment")
                print("3. Xem biểu đồ experiment")
                print("4. Xem ảnh AI experiment")
                print("5. So sánh experiments")
                print("6. Export experiment")
                print("7. Quay lại menu chính")
                
                exp_choice = input("\nChọn chức năng (1-7): ").strip()
                
                if exp_choice == "1":
                    app.list_experiments()
                elif exp_choice == "2":
                    app.view_experiment()
                elif exp_choice == "3":
                    app.show_experiment_plots()
                elif exp_choice == "4":
                    app.show_experiment_images()
                elif exp_choice == "5":
                    app.compare_experiments()
                elif exp_choice == "6":
                    app.export_experiment()
                elif exp_choice == "7":
                    break
                else:
                    print("❌ Lựa chọn không hợp lệ!")
            
        elif choice == "8":
            # Tiếp tục RL Training từ experiment trước
            print("\n🔄 TIẾP TỤC RL TRAINING")
            print("=" * 50)
            
            # Liệt kê experiments có sẵn
            experiments = app.experiment_viewer.list_all_experiments()
            if not experiments:
                print("❌ Không có experiment nào để tiếp tục")
                continue
            
            print("\nCác experiments có sẵn:")
            for i, exp in enumerate(experiments[:5], 1):  # Hiển thị 5 experiments gần nhất
                print(f"{i}. {exp['experiment_id']} - {exp.get('status', 'Unknown')}")
            
            try:
                exp_choice = input("\nChọn experiment để tiếp tục (1-5, hoặc Enter để bỏ qua): ").strip()
                if exp_choice and exp_choice.isdigit():
                    idx = int(exp_choice) - 1
                    if 0 <= idx < len(experiments):
                        selected_exp = experiments[idx]
                        exp_dir = selected_exp.get('path', '')
                        
                        if exp_dir and os.path.exists(exp_dir):
                            print(f"\n🔄 Tiếp tục training từ: {selected_exp['experiment_id']}")
                            
                            epochs = input("Nhập số epochs để tiếp tục (mặc định 3): ").strip()
                            epochs = int(epochs) if epochs.isdigit() else 3
                            
                            # Chạy RL training với continue_from
                            results = app.rl_enhancement.run_reinforcement_learning(
                                epochs=epochs, 
                                continue_from=exp_dir
                            )
                            
                            if results:
                                print(f"\n✅ Tiếp tục training hoàn tất!")
                                print(f"🎯 Final Reward: {results.get('reward', 0):.4f}")
                                print(f"📈 Total AI Images: {results.get('total_ai_images', 0)}")
                            else:
                                print("❌ Tiếp tục training thất bại")
                        else:
                            print("❌ Experiment directory không tồn tại")
                    else:
                        print("❌ Lựa chọn không hợp lệ")
                else:
                    print("Đã hủy tiếp tục training")
            except Exception as e:
                print(f"❌ Lỗi: {e}")
            
        elif choice == "9":
            print("👋 Tạm biệt!")
            break
            
        else:
            print("❌ Lựa chọn không hợp lệ!")

if __name__ == "__main__":
    main()
