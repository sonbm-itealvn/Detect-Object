import json
import subprocess
import threading
import re
import tkinter as tk
from tkinter import Entry, filedialog, Label, Button, Canvas, Frame, Scrollbar, Text
from PIL import Image, ImageTk, ImageDraw
import os
import glob
from sentence_transformers import SentenceTransformer, util

from RL.rl_enhancement import AppReinforcementLearning
from RL.training_evaluator import TrainingEvaluator

class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Detection & Relationship Analysis")
        self.root.geometry("1400x800")
        self.root.configure(bg="#f5f5f5")

        # Tiêu đề chính
        title_frame = Frame(root, bg="#2c3e50", height=80)
        title_frame.pack(fill="x", pady=(0, 20))
        title_frame.pack_propagate(False)
        
        self.title_label = Label(title_frame, text="🔍 Object Detection & Relationship Analysis", 
                               font=("Arial", 18, "bold"), bg="#2c3e50", fg="white")
        self.title_label.pack(expand=True)

        # Frame chứa các nút điều khiển
        control_frame = Frame(root, bg="#f5f5f5")
        control_frame.pack(pady=(0, 20))

        self.btn_select = Button(control_frame, text="📁 Select Image", command=self.select_image, 
                               font=("Arial", 12, "bold"), bg="#3498db", fg="white", 
                               width=15, height=2, relief="flat", bd=0)
        self.btn_select.pack(side="left", padx=10)

        self.btn_run = Button(control_frame, text="▶️ Detect Object", command=self.run_pipeline_thread, 
                            font=("Arial", 12, "bold"), bg="#27ae60", fg="white", 
                            width=15, height=2, relief="flat", bd=0)
        self.btn_run.pack(side="left", padx=10)

        self.btn_refresh = Button(control_frame, text="🔄 Reload Data", command=self.refresh_data, 
                                font=("Arial", 12, "bold"), bg="#f39c12", fg="white", 
                                width=15, height=2, relief="flat", bd=0)
        self.btn_refresh.pack(side="left", padx=10)
        
        self.rl_enhancement = AppReinforcementLearning(self)
        self.training_evaluator = TrainingEvaluator()

        self.btn_rl_train = Button(control_frame, text="🧠 RL Training", 
                                 command=self.run_rl_training,
                                 font=("Arial", 12, "bold"), bg="#9b59b6", fg="white", 
                                 width=15, height=2, relief="flat", bd=0)
        self.btn_rl_train.pack(side="left", padx=10)
        
        self.btn_generate_synthetic = Button(control_frame, text="🎨 Generate Synthetic", 
                                           command=self.generate_synthetic_data,
                                           font=("Arial", 12, "bold"), bg="#e67e22", fg="white", 
                                           width=15, height=2, relief="flat", bd=0)
        self.btn_generate_synthetic.pack(side="left", padx=10)
        
        self.btn_evaluate_training = Button(control_frame, text="📊 Evaluate Training", 
                                          command=self.evaluate_training_results,
                                          font=("Arial", 12, "bold"), bg="#8e44ad", fg="white", 
                                          width=15, height=2, relief="flat", bd=0)
        self.btn_evaluate_training.pack(side="left", padx=10)

        # Frame chính chứa 3 cột
        main_frame = Frame(root, bg="#f5f5f5")
        main_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        # Cột 1: Hiển thị ảnh
        self.image_frame = Frame(main_frame, bg="white", relief="solid", bd=2)
        self.image_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        image_title = Label(self.image_frame, text="🖼️ Hình ảnh", font=("Arial", 14, "bold"), 
                           bg="white", fg="#2c3e50")
        image_title.pack(pady=10)
        
        self.canvas = Canvas(self.image_frame, width=500, height=400, bg="white", relief="flat")
        self.canvas.pack(pady=(0, 10), padx=10)

        # Cột 2: Danh sách vật thể
        self.objects_frame = Frame(main_frame, bg="white", relief="solid", bd=2, width=300)
        self.objects_frame.pack(side="left", fill="y", padx=(0, 10))
        self.objects_frame.pack_propagate(False)
        
        objects_title = Label(self.objects_frame, text="📦 Vật thể được phát hiện", 
                            font=("Arial", 14, "bold"), bg="white", fg="#2c3e50")
        objects_title.pack(pady=10)
        
        # Scrollbar cho danh sách vật thể
        objects_scroll_frame = Frame(self.objects_frame, bg="white")
        objects_scroll_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        self.objects_text = Text(objects_scroll_frame, height=15, width=35, font=("Arial", 10), 
                               bg="#f8f9fa", fg="#2c3e50", relief="flat", bd=0, wrap="word")
        objects_scrollbar = Scrollbar(objects_scroll_frame, orient="vertical", command=self.objects_text.yview)
        self.objects_text.configure(yscrollcommand=objects_scrollbar.set)
        
        self.objects_text.pack(side="left", fill="both", expand=True)
        objects_scrollbar.pack(side="right", fill="y")

        # Cột 3: Danh sách mối quan hệ
        self.relationships_frame = Frame(main_frame, bg="white", relief="solid", bd=2, width=300)
        self.relationships_frame.pack(side="left", fill="y")
        self.relationships_frame.pack_propagate(False)
        
        relationships_title = Label(self.relationships_frame, text="🔗 Mối quan hệ", 
                                  font=("Arial", 14, "bold"), bg="white", fg="#2c3e50")
        relationships_title.pack(pady=10)
        
        # Scrollbar cho danh sách mối quan hệ
        relationships_scroll_frame = Frame(self.relationships_frame, bg="white")
        relationships_scroll_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        self.relationships_text = Text(relationships_scroll_frame, height=15, width=35, font=("Arial", 10), 
                                     bg="#f8f9fa", fg="#2c3e50", relief="flat", bd=0, wrap="word")
        relationships_scrollbar = Scrollbar(relationships_scroll_frame, orient="vertical", command=self.relationships_text.yview)
        self.relationships_text.configure(yscrollcommand=relationships_scrollbar.set)
        
        self.relationships_text.pack(side="left", fill="both", expand=True)
        relationships_scrollbar.pack(side="right", fill="y")

        # Các đường dẫn mặc định
        self.image_path = None
        self.result_image_path = "result.jpg"
        self.result_json_path = "converted_bboxes.json"
        self.relationship_json_path = "relationships.json"
        self.checkpoint_path = "reltr_finetuned.pth" 

        # Load model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def select_image(self):
        file_path = filedialog.askopenfilename(title="Chọn ảnh", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.image_path = file_path
            self.display_image(file_path)
            # Tự động tải lại dữ liệu JSON khi chọn ảnh mới
            self.title_label.config(text="🔄 Đang tải dữ liệu JSON...")
            self.load_and_display_objects()
            self.load_and_display_relationships()
            self.title_label.config(text="✅ Đã tải dữ liệu JSON cho ảnh mới!")

    def display_image(self, path):
        try:
            image = Image.open(path)
            # Tính toán kích thước phù hợp với canvas
            canvas_width = 500
            canvas_height = 400
            
            # Tính tỷ lệ để giữ nguyên tỷ lệ ảnh
            img_width, img_height = image.size
            ratio = min(canvas_width/img_width, canvas_height/img_height)
            
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)
            
            image = image.resize((new_width, new_height), Image.LANCZOS)
            self.img_tk = ImageTk.PhotoImage(image)
            
            # Xóa ảnh cũ và vẽ ảnh mới ở giữa canvas
            self.canvas.delete("all")
            x = canvas_width // 2
            y = canvas_height // 2
            self.canvas.create_image(x, y, image=self.img_tk)
            
        except Exception as e:
            print(f"❌ Lỗi hiển thị ảnh: {e}")
            self.canvas.delete("all")
            self.canvas.create_text(250, 200, text=f"❌ Lỗi tải ảnh:\n{str(e)}", 
                                  font=("Arial", 12), fill="red", justify="center")


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
                self.objects_text.delete(1.0, tk.END)
                self.objects_text.insert(tk.END, "❌ Không có vật thể nào được phát hiện")
                return

            # Xóa nội dung cũ
            self.objects_text.delete(1.0, tk.END)
            
            # Hiển thị thông tin vật thể
            for i, obj in enumerate(objects, 1):
                class_name = obj.get("class", "Unknown")
                bbox = obj.get("bbox", [])
                
                if len(bbox) >= 4:
                    x, y, w, h = bbox[:4]
                    info = f"🔸 {i}. {class_name.upper()}\n"
                    info += f"   📍 Vị trí: ({x}, {y})\n"
                    info += f"   📏 Kích thước: {w-x} x {h-y}\n"
                    info += f"   🎯 Độ tin cậy: {obj.get('confidence', 'N/A')}\n\n"
                else:
                    info = f"🔸 {i}. {class_name.upper()}\n"
                    info += f"   📍 Thông tin bbox không hợp lệ\n\n"
                
                self.objects_text.insert(tk.END, info)
                
        except FileNotFoundError:
            self.objects_text.delete(1.0, tk.END)
            self.objects_text.insert(tk.END, f"❌ Không tìm thấy file: {self.result_json_path}")
        except json.JSONDecodeError:
            self.objects_text.delete(1.0, tk.END)
            self.objects_text.insert(tk.END, "❌ Lỗi đọc file JSON")
        except Exception as e:
            self.objects_text.delete(1.0, tk.END)
            self.objects_text.insert(tk.END, f"❌ Lỗi: {str(e)}")

    def load_and_display_relationships(self):
        """Tải và hiển thị danh sách mối quan hệ từ JSON"""
        try:
            with open(self.relationship_json_path, "r", encoding="utf-8") as f:
                relationships = json.load(f)
            
            if not relationships:
                self.relationships_text.delete(1.0, tk.END)
                self.relationships_text.insert(tk.END, "❌ Không có mối quan hệ nào được phát hiện")
                return

            # Xóa nội dung cũ
            self.relationships_text.delete(1.0, tk.END)
            
            # Hiển thị thông tin mối quan hệ
            for i, rel in enumerate(relationships, 1):
                subject = rel.get("subject", "Unknown")
                relation = rel.get("relation", "Unknown")
                obj = rel.get("object", "Unknown")
                similarity = rel.get("visual_similarity", 0)
                
                # Màu sắc dựa trên độ tin cậy (nếu có visual_similarity)
                if similarity > 0:
                    confidence_color = "🟢" if similarity > 0.7 else "🟡" if similarity > 0.4 else "🔴"
                    info = f"{confidence_color} {i}. {subject.upper()}\n"
                    info += f"   🔗 {relation.upper()}\n"
                    info += f"   🎯 {obj.upper()}\n"
                    info += f"   📊 Độ tin cậy: {similarity:.2f}\n\n"
                else:
                    # Nếu không có visual_similarity, hiển thị đơn giản
                    info = f"🔸 {i}. {subject.upper()}\n"
                    info += f"   🔗 {relation.upper()}\n"
                    info += f"   🎯 {obj.upper()}\n\n"
                
                self.relationships_text.insert(tk.END, info)
                
        except FileNotFoundError:
            self.relationships_text.delete(1.0, tk.END)
            self.relationships_text.insert(tk.END, f"❌ Không tìm thấy file: {self.relationship_json_path}")
        except json.JSONDecodeError:
            self.relationships_text.delete(1.0, tk.END)
            self.relationships_text.insert(tk.END, "❌ Lỗi đọc file JSON")
        except Exception as e:
            self.relationships_text.delete(1.0, tk.END)
            self.relationships_text.insert(tk.END, f"❌ Lỗi: {str(e)}")

    def refresh_data(self):
        """Tải lại dữ liệu JSON mà không cần chạy lại pipeline"""
        self.title_label.config(text="🔄 Đang tải lại dữ liệu...")
        self.load_and_display_objects()
        self.load_and_display_relationships()
        self.title_label.config(text="✅ Đã tải lại dữ liệu thành công!")

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
            
            # Hiển thị ảnh mới
            self.display_image(result_path)
            print(f"✅ Đã vẽ bbox mối quan hệ và lưu tại: {result_path}")
            
        except Exception as e:
            print(f"❌ Lỗi khi vẽ bbox mối quan hệ: {e}")
            self.title_label.config(text=f"❌ Lỗi vẽ bbox: {e}")

    def draw_relationship_boxes(self, subject_name, object_name):
        try:
            with open(self.result_json_path, "r") as f:
                data = json.load(f)

            print("✅ JSON data loaded:", data)

            if isinstance(data, list):
                data = data[0]

            objects = data.get("objects", [])

            print("🔍 Danh sách objects:", [obj.get("class", "") for obj in objects])
            print("🔍 Subject cần tìm:", subject_name, "| Object cần tìm:", object_name)

            # Tìm subject
            subject_box = next(
                (obj for obj in objects if obj.get("class", "").lower() == subject_name.lower()), None
            )

            # Tìm object nếu có
            object_box = None
            if object_name:
                object_box = next(
                    (obj for obj in objects if obj.get("class", "").lower() == object_name.lower()), None
                )

            if not subject_box:
                self.label.config(text=f"❌ Không tìm thấy subject: {subject_name} trong JSON!")
                print("❌ Lỗi tìm subject:", subject_name)
                return

            if object_name and not object_box:
                self.label.config(text=f"❌ Không tìm thấy object: {object_name} trong JSON!")
                print("❌ Lỗi tìm object:", object_name)
                return

            image = Image.open(self.image_path)
            draw = ImageDraw.Draw(image)

            # Vẽ subject với box đậm hơn và chữ to hơn
            sx, sy, sw, sh = subject_box["bbox"]
            draw.rectangle([sx, sy, sw, sh], outline="red", width=6)
            try:
                from PIL import ImageFont
                font = ImageFont.truetype("arial.ttf", 24)
            except:
                font = ImageFont.load_default()
            draw.text((sx, sy - 30), subject_name, fill="red", font=font)

            # Vẽ object nếu có với box đậm hơn và chữ to hơn
            if object_box:
                ox, oy, ow, oh = object_box["bbox"]
                draw.rectangle([ox, oy, ow, oh], outline="blue", width=6)
                try:
                    from PIL import ImageFont
                    font = ImageFont.truetype("arial.ttf", 24)
                except:
                    font = ImageFont.load_default()
                draw.text((ox, oy - 30), object_name, fill="blue", font=font)

            result_path = "relationship_result.jpg"
            image.save(result_path)
            self.display_image(result_path)

            self.label.config(text="✅ Đã vẽ xong box!")

        except Exception as e:
            self.label.config(text=f"❌ Lỗi khi vẽ box: {e}")
            print(f"❌ Lỗi khi vẽ box: {e}")

    def run_pipeline_thread(self):
        thread = threading.Thread(target=self.run_pipeline)
        thread.start()

    def run_pipeline(self):
        if not self.image_path:
            self.title_label.config(text="❌ Hãy chọn ảnh trước!")
            return

        self.title_label.config(text="⏳ Đang xử lý... Vui lòng chờ.")

        try:
            # 1️⃣ Chạy detect_objects.py
            self.title_label.config(text="🔍 Đang phát hiện vật thể...")
            detect_thread = threading.Thread(target=subprocess.run, args=(["python", "detect_objects.py", self.image_path],))
            detect_thread.start()
            detect_thread.join()  # Đợi detect_objects.py chạy xong

            # 2️⃣ Chạy convert_yolo_to_reltr.py (sau khi detect_objects.py hoàn tất)
            self.title_label.config(text="🔄 Đang chuyển đổi dữ liệu YOLO...")
            convert_thread = threading.Thread(target=subprocess.run, args=(["python", "convert_yolo_to_reltr.py", "result.json"],))
            convert_thread.start()
            convert_thread.join()  # Đợi convert_yolo_to_reltr.py chạy xong

            # 3️⃣ Chạy boundingbox_objects.py (sau khi convert_yolo_to_reltr.py hoàn tất)
            self.title_label.config(text="🔗 Đang xác định mối quan hệ giữa các vật thể...")
            boundingbox_thread = threading.Thread(target=subprocess.run, args=(["python", "boundingbox_objects.py", "--yolo_json", self.result_json_path,"--img_path",self.image_path,"--device","cpu", "--resume", self.checkpoint_path],))
            boundingbox_thread.start()
            boundingbox_thread.join()  # Đợi boundingbox_objects.py chạy xong

            image_dir = os.path.dirname(self.image_path)
            image_id = os.path.splitext(os.path.basename(self.image_path))[0]

            # ✅ Tìm ảnh output_anh2.jpg ở bất kỳ thư mục nào
            output_images = glob.glob(f"**/output_{image_id}.jpg", recursive=True)

            if output_images:
                latest_result = max(output_images, key=os.path.getmtime)  # Lấy ảnh mới nhất nếu có nhiều ảnh trùng tên
                self.display_image(latest_result)
                self.title_label.config(text="✅ Hoàn tất! Đây là kết quả.")
            else:
                print("📂 Danh sách file trong thư mục:", os.listdir(image_dir))  # Debug kiểm tra
                self.title_label.config(text="❌ Không tìm thấy ảnh kết quả!")

            # 4️⃣ Tải và hiển thị dữ liệu JSON
            self.title_label.config(text="📊 Đang tải dữ liệu kết quả...")
            self.load_and_display_objects()
            self.load_and_display_relationships()
            
            # 5️⃣ Vẽ bbox mối quan hệ trên ảnh
            self.title_label.config(text="🎨 Đang vẽ bbox mối quan hệ...")
            self.draw_relationship_boxes_on_image()
            self.title_label.config(text="✅ Hoàn tất! Dữ liệu đã được tải và vẽ bbox.")
            
        except Exception as e:
            self.title_label.config(text=f"❌ Lỗi: {e}")
            print(f"❌ Lỗi xảy ra: {e}")

    def run_rl_training(self):
        """Run reinforcement learning training in separate thread"""
        self.title_label.config(text="🧠 Đang chạy RL Training...")
        
        def rl_training_thread():
            try:
                results = self.rl_enhancement.run_reinforcement_learning()
                self.title_label.config(text=f"✅ RL Training hoàn tất! Reward: {results['reward']:.3f}")
            except Exception as e:
                self.title_label.config(text=f"❌ RL Training lỗi: {e}")
                print(f"❌ RL Training error: {e}")
        
        # Chạy RL training trong thread riêng để không block UI
        thread = threading.Thread(target=rl_training_thread)
        thread.daemon = True  # Thread sẽ tự động kết thúc khi app đóng
        thread.start()
    
    def generate_synthetic_data(self):
        """Generate synthetic dataset from relationships in separate thread"""
        self.title_label.config(text="🎨 Đang tạo dữ liệu synthetic...")
        
        def synthetic_generation_thread():
            try:
                synthetic_data = self.rl_enhancement.generate_synthetic_dataset()
                self.title_label.config(text=f"✅ Đã tạo {len(synthetic_data)} ảnh synthetic!")
            except Exception as e:
                self.title_label.config(text=f"❌ Tạo synthetic data lỗi: {e}")
                print(f"❌ Synthetic data generation error: {e}")
        
        # Chạy synthetic data generation trong thread riêng
        thread = threading.Thread(target=synthetic_generation_thread)
        thread.daemon = True
        thread.start()
    
    def evaluate_training_results(self):
        """Evaluate training results and create comprehensive report"""
        self.title_label.config(text="📊 Đang đánh giá kết quả training...")
        
        def evaluation_thread():
            try:
                # Create comprehensive evaluation report
                report = self.training_evaluator.create_comprehensive_report()
                
                if 'error' in report:
                    self.title_label.config(text=f"❌ {report['error']}")
                    return
                
                # Display evaluation results
                total_sessions = report.get('total_sessions', 0)
                overall_analysis = report.get('overall_analysis', {})
                
                if total_sessions > 0:
                    avg_reward = overall_analysis.get('average_reward', 0)
                    total_images = overall_analysis.get('total_ai_images_generated', 0)
                    
                    self.title_label.config(text=f"✅ Đánh giá hoàn tất! {total_sessions} sessions, Reward: {avg_reward:.3f}, AI Images: {total_images}")
                    
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
                    self.title_label.config(text="❌ Không tìm thấy kết quả training để đánh giá")
                    
            except Exception as e:
                self.title_label.config(text=f"❌ Lỗi đánh giá: {e}")
                print(f"❌ Evaluation error: {e}")
        
        # Chạy evaluation trong thread riêng
        thread = threading.Thread(target=evaluation_thread)
        thread.daemon = True
        thread.start()

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()
