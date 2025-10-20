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
            objects_dict = {}
            for obj in objects:
                class_name = obj.get("class", "").lower()
                objects_dict[class_name] = obj
            
            # Vẽ bbox cho từng mối quan hệ
            colors = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "cyan"]
            
            for i, rel in enumerate(relationships_data):
                subject = rel.get("subject", "").lower()
                obj = rel.get("object", "").lower()
                
                color = colors[i % len(colors)]
                
                # Vẽ bbox cho subject
                if subject in objects_dict:
                    bbox = objects_dict[subject].get("bbox", [])
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
                if obj in objects_dict:
                    bbox = objects_dict[obj].get("bbox", [])
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

    def run_rl_training(self):
        """Run reinforcement learning training"""
        print("🧠 Đang chạy RL Training...")
        
        try:
            results = self.rl_enhancement.run_reinforcement_learning()
            print(f"✅ RL Training hoàn tất! Reward: {results['reward']:.3f}")
        except Exception as e:
            print(f"❌ RL Training lỗi: {e}")
    
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
        print("7. Thoát")
        
        choice = input("\nNhập lựa chọn (1-7): ").strip()
        
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
            app.run_rl_training()
            
        elif choice == "5":
            app.generate_synthetic_data()
            
        elif choice == "6":
            app.evaluate_training_results()
            
        elif choice == "7":
            print("👋 Tạm biệt!")
            break
            
        else:
            print("❌ Lựa chọn không hợp lệ!")

if __name__ == "__main__":
    main()
