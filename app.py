import json
import subprocess
import threading
import re
import tkinter as tk
from tkinter import Entry, filedialog, Label, Button, Canvas, Frame, Scrollbar, Text
from PIL import Image, ImageTk, ImageDraw
import os
import sys
import glob
import cv2
from typing import Optional
from sentence_transformers import SentenceTransformer, util

from RL.rl_enhancement import AppReinforcementLearning
from RL.training_evaluator import TrainingEvaluator
from video_relation_pipeline import VideoRelationPipeline

class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Detection & Relationship Analysis")
        self.root.geometry("1400x800")
        self.root.minsize(800, 600)
        
        # Modern color scheme
        self.colors = {
            'bg_main': '#0f172a',  # Dark slate
            'bg_secondary': '#1e293b',  # Slate
            'bg_card': '#ffffff',
            'bg_card_dark': '#f8fafc',
            'accent_primary': '#3b82f6',  # Blue
            'accent_success': '#10b981',  # Green
            'accent_warning': '#f59e0b',  # Amber
            'accent_danger': '#ef4444',  # Red
            'accent_purple': '#8b5cf6',  # Purple
            'accent_teal': '#14b8a6',  # Teal
            'text_primary': '#1e293b',
            'text_secondary': '#64748b',
            'text_light': '#ffffff',
            'border': '#e2e8f0',
            'shadow': '#cbd5e1'
        }
        
        self.root.configure(bg=self.colors['bg_main'])
        
        # Bind resize event để responsive
        self.root.bind("<Configure>", self.on_window_resize)

        # Modern header với gradient effect
        title_frame = Frame(root, bg=self.colors['bg_secondary'], height=70)
        title_frame.pack(fill="x", pady=(0, 15))
        title_frame.pack_propagate(False)
        
        # Title với better styling
        title_inner = Frame(title_frame, bg=self.colors['bg_secondary'])
        title_inner.pack(expand=True, fill="both", padx=20)
        
        self.title_label = Label(title_inner, 
                               text="🔍 Object Detection & Relationship Analysis", 
                               font=("Segoe UI", 16, "bold"), 
                               bg=self.colors['bg_secondary'], 
                               fg=self.colors['text_light'])
        self.title_label.pack(expand=True, pady=15)

        # Modern control panel với card design
        control_container = Frame(root, bg=self.colors['bg_main'])
        control_container.pack(fill="x", pady=(0, 15), padx=15)
        
        # Card frame cho buttons
        control_card = Frame(control_container, bg=self.colors['bg_card'], relief="flat", bd=0)
        control_card.pack(fill="x", padx=0, pady=0)
        
        # Canvas và scrollbar cho control buttons
        control_canvas = Canvas(control_card, bg=self.colors['bg_card'], height=90, highlightthickness=0)
        control_scrollbar = Scrollbar(control_card, orient="horizontal", command=control_canvas.xview,
                                     bg=self.colors['bg_card'], troughcolor=self.colors['bg_card_dark'],
                                     activebackground=self.colors['accent_primary'])
        control_scrollable_frame = Frame(control_canvas, bg=self.colors['bg_card'])
        
        control_scrollable_frame.bind(
            "<Configure>",
            lambda e: control_canvas.configure(scrollregion=control_canvas.bbox("all"))
        )
        
        control_canvas.create_window((0, 0), window=control_scrollable_frame, anchor="nw")
        control_canvas.configure(xscrollcommand=control_scrollbar.set)
        
        control_canvas.pack(side="left", fill="both", expand=True, padx=15, pady=15)
        control_scrollbar.pack(side="right", fill="y", padx=(0, 15), pady=15)
        
        # Lưu reference để có thể cập nhật
        self.control_frame = control_scrollable_frame
        self.control_canvas = control_canvas
        
        self.rl_enhancement = AppReinforcementLearning(self)
        self.training_evaluator = TrainingEvaluator()

        # Modern button style với hover effects
        def create_modern_button(parent, text, command, bg_color, hover_color=None):
            if hover_color is None:
                hover_color = bg_color
            btn = Button(parent, text=text, command=command,
                        font=("Segoe UI", 10, "bold"), 
                        bg=bg_color, fg="white", 
                        width=14, height=2, 
                        relief="flat", bd=0, 
                        cursor="hand2",
                        activebackground=hover_color,
                        activeforeground="white")
            
            # Hover effect
            def on_enter(e):
                btn.configure(bg=hover_color)
            def on_leave(e):
                btn.configure(bg=bg_color)
            
            btn.bind("<Enter>", on_enter)
            btn.bind("<Leave>", on_leave)
            return btn

        # Tạo các nút với modern styling
        self.btn_select = create_modern_button(
            self.control_frame, "📁 Select Image", self.select_image,
            self.colors['accent_primary'], '#2563eb'
        )
        self.btn_select.pack(side="left", padx=8, pady=8)

        self.btn_run = create_modern_button(
            self.control_frame, "▶️ Detect Object", self.run_pipeline_thread,
            self.colors['accent_success'], '#059669'
        )
        self.btn_run.pack(side="left", padx=8, pady=8)

        self.btn_refresh = create_modern_button(
            self.control_frame, "🔄 Reload Data", self.refresh_data,
            self.colors['accent_warning'], '#d97706'
        )
        self.btn_refresh.pack(side="left", padx=8, pady=8)

        self.btn_rl_train = create_modern_button(
            self.control_frame, "🧠 RL Training", self.run_rl_training,
            self.colors['accent_purple'], '#7c3aed'
        )
        self.btn_rl_train.pack(side="left", padx=8, pady=8)
        
        self.btn_generate_synthetic = create_modern_button(
            self.control_frame, "🎨 Generate Synthetic", self.generate_synthetic_data,
            '#f97316', '#ea580c'  # Orange
        )
        self.btn_generate_synthetic.pack(side="left", padx=8, pady=8)
        
        self.btn_evaluate_training = create_modern_button(
            self.control_frame, "📊 Evaluate Training", self.evaluate_training_results,
            '#a855f7', '#9333ea'  # Purple variant
        )
        self.btn_evaluate_training.pack(side="left", padx=8, pady=8)

        self.btn_select_video = create_modern_button(
            self.control_frame, "📹 Select Video", self.select_video,
            self.colors['accent_teal'], '#0d9488'
        )
        self.btn_select_video.pack(side="left", padx=8, pady=8)

        self.btn_run_video = create_modern_button(
            self.control_frame, "▶️ Run Video Demo", self.run_video_demo_thread,
            '#06b6d4', '#0891b2'  # Cyan
        )
        self.btn_run_video.pack(side="left", padx=8, pady=8)

        self.btn_stop_video = create_modern_button(
            self.control_frame, "⏹️ Stop Video", self.stop_video_demo,
            self.colors['accent_danger'], '#dc2626'
        )
        self.btn_stop_video.pack(side="left", padx=8, pady=8)
        
        # Update scroll region
        self.control_frame.update_idletasks()
        self.control_canvas.configure(scrollregion=self.control_canvas.bbox("all"))

        # Frame chính chứa 3 cột với modern card design
        main_container = Frame(root, bg=self.colors['bg_main'])
        main_container.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        
        # Sử dụng grid layout để responsive
        main_container.grid_columnconfigure(0, weight=2, minsize=350)
        main_container.grid_columnconfigure(1, weight=1, minsize=250)
        main_container.grid_columnconfigure(2, weight=1, minsize=250)
        main_container.grid_rowconfigure(0, weight=1)

        # Cột 1: Hiển thị ảnh với modern card
        image_card = Frame(main_container, bg=self.colors['bg_card'], relief="flat", bd=0)
        image_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        image_card.grid_propagate(False)
        
        # Header cho image card
        image_header = Frame(image_card, bg=self.colors['accent_primary'], height=45)
        image_header.pack(fill="x")
        image_header.pack_propagate(False)
        
        image_title = Label(image_header, text="🖼️ Hình ảnh", 
                          font=("Segoe UI", 13, "bold"), 
                          bg=self.colors['accent_primary'], 
                          fg="white")
        image_title.pack(expand=True, pady=12)
        
        # Canvas container với padding
        canvas_container = Frame(image_card, bg=self.colors['bg_card'])
        canvas_container.pack(fill="both", expand=True, padx=15, pady=15)
        
        self.canvas = Canvas(canvas_container, bg="#f1f5f9", relief="flat", 
                           highlightthickness=1, highlightbackground=self.colors['border'],
                           highlightcolor=self.colors['accent_primary'])
        self.canvas.pack(fill="both", expand=True)

        # Cột 2: Danh sách vật thể với modern card
        objects_card = Frame(main_container, bg=self.colors['bg_card'], relief="flat", bd=0)
        objects_card.grid(row=0, column=1, sticky="nsew", padx=(0, 10))
        objects_card.grid_propagate(False)
        
        # Header cho objects card
        objects_header = Frame(objects_card, bg=self.colors['accent_success'], height=45)
        objects_header.pack(fill="x")
        objects_header.pack_propagate(False)
        
        objects_title = Label(objects_header, text="📦 Vật thể được phát hiện", 
                            font=("Segoe UI", 13, "bold"), 
                            bg=self.colors['accent_success'], 
                            fg="white")
        objects_title.pack(expand=True, pady=12)
        
        # Scrollbar cho danh sách vật thể
        objects_scroll_frame = Frame(objects_card, bg=self.colors['bg_card'])
        objects_scroll_frame.pack(fill="both", expand=True, padx=12, pady=12)
        
        self.objects_text = Text(objects_scroll_frame, 
                               font=("Segoe UI", 10), 
                               bg="#f8fafc", 
                               fg=self.colors['text_primary'],
                               relief="flat", 
                               bd=0, 
                               wrap="word",
                               padx=10,
                               pady=10)
        objects_scrollbar = Scrollbar(objects_scroll_frame, 
                                     orient="vertical", 
                                     command=self.objects_text.yview,
                                     bg=self.colors['bg_card'],
                                     troughcolor=self.colors['bg_card_dark'],
                                     activebackground=self.colors['accent_success'])
        self.objects_text.configure(yscrollcommand=objects_scrollbar.set)
        
        self.objects_text.pack(side="left", fill="both", expand=True)
        objects_scrollbar.pack(side="right", fill="y")

        # Cột 3: Danh sách mối quan hệ với modern card
        relationships_card = Frame(main_container, bg=self.colors['bg_card'], relief="flat", bd=0)
        relationships_card.grid(row=0, column=2, sticky="nsew")
        relationships_card.grid_propagate(False)
        
        # Header cho relationships card
        relationships_header = Frame(relationships_card, bg=self.colors['accent_purple'], height=45)
        relationships_header.pack(fill="x")
        relationships_header.pack_propagate(False)
        
        relationships_title = Label(relationships_header, text="🔗 Mối quan hệ", 
                                  font=("Segoe UI", 13, "bold"), 
                                  bg=self.colors['accent_purple'], 
                                  fg="white")
        relationships_title.pack(expand=True, pady=12)
        
        # Scrollbar cho danh sách mối quan hệ
        relationships_scroll_frame = Frame(relationships_card, bg=self.colors['bg_card'])
        relationships_scroll_frame.pack(fill="both", expand=True, padx=12, pady=12)
        
        self.relationships_text = Text(relationships_scroll_frame, 
                                     font=("Segoe UI", 10), 
                                     bg="#f8fafc", 
                                     fg=self.colors['text_primary'],
                                     relief="flat", 
                                     bd=0, 
                                     wrap="word",
                                     padx=10,
                                     pady=10)
        relationships_scrollbar = Scrollbar(relationships_scroll_frame, 
                                         orient="vertical", 
                                         command=self.relationships_text.yview,
                                         bg=self.colors['bg_card'],
                                         troughcolor=self.colors['bg_card_dark'],
                                         activebackground=self.colors['accent_purple'])
        self.relationships_text.configure(yscrollcommand=relationships_scrollbar.set)
        
        self.relationships_text.pack(side="left", fill="both", expand=True)
        relationships_scrollbar.pack(side="right", fill="y")
        
        # Lưu reference cho responsive
        self.main_container = main_container
        self.image_frame = image_card  # Update reference
        self.objects_frame = objects_card  # Update reference
        self.relationships_frame = relationships_card  # Update reference

    def on_window_resize(self, event=None):
        """Xử lý khi window resize để responsive"""
        if event and event.widget == self.root:
            # Cập nhật scroll region cho control buttons
            try:
                self.control_frame.update_idletasks()
                self.control_canvas.configure(scrollregion=self.control_canvas.bbox("all"))
            except:
                pass
            
            # Cập nhật canvas size nếu có ảnh
            if hasattr(self, 'img_tk') and hasattr(self, '_original_image'):
                # Delay một chút để canvas có thời gian resize
                self.root.after(100, self._resize_canvas_image)

    def _resize_canvas_image(self):
        """Resize ảnh trên canvas khi window resize"""
        if not hasattr(self, 'img_tk') or not hasattr(self, 'canvas'):
            return
        
        try:
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                return
            
            # Lấy ảnh gốc nếu có
            if hasattr(self, '_original_image'):
                image = self._original_image
            else:
                return
            
            img_width, img_height = image.size
            if img_width <= 0 or img_height <= 0:
                return
            
            ratio = min(canvas_width / img_width, canvas_height / img_height)
            new_width = max(1, int(img_width * ratio))
            new_height = max(1, int(img_height * ratio))
            
            resized = image.resize((new_width, new_height), Image.LANCZOS)
            self.img_tk = ImageTk.PhotoImage(resized)
            self.canvas.delete("all")
            self.canvas.create_image(canvas_width // 2, canvas_height // 2, image=self.img_tk, anchor="center")
        except Exception as e:
            print(f"Error resizing canvas image: {e}")

        # Các đường dẫn mặc định
        self.image_path = None
        self.result_image_path = "result.jpg"
        self.result_json_path = "converted_bboxes.json"
        self.relationship_json_path = "relationships.json"
        self.checkpoint_path = "reltr_finetuned.pth" 

        # Load model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.video_path = None
        self.video_pipeline: Optional[VideoRelationPipeline] = None
        self.video_thread: Optional[threading.Thread] = None
        self.video_stop_event = threading.Event()
        self.latest_video_outputs = {}

    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")],
        )
        if not file_path:
            return
        self.image_path = file_path
        self.display_image(file_path)
        self.title_label.config(text="Loading JSON data...")
        self.load_and_display_objects()
        self.load_and_display_relationships()
        self.title_label.config(text="JSON data loaded for the new image.")

    def select_video(self):
        file_path = filedialog.askopenfilename(
            title="Select video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")],
        )
        if not file_path:
            return
        self.video_path = file_path
        self.title_label.config(text=f"Selected video: {os.path.basename(file_path)}")
        self.display_video_thumbnail(file_path)

    def display_video_thumbnail(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        success, frame = cap.read()
        cap.release()
        if success:
            self.display_frame_from_array(frame)
            self.title_label.config(text=f"Video ready: {os.path.basename(video_path)}")
        else:
            self.title_label.config(text="Unable to read the first frame from the video.")

    def display_image(self, path: str):
        try:
            image = Image.open(path)
        except Exception as exc:
            print(f"Error loading image: {exc}")
            self._show_error_on_canvas(f"Error loading image:\\n{exc}")
            return
        self._show_image_on_canvas(image)

    def display_frame_from_array(self, frame):
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            self._show_image_on_canvas(image)
        except Exception as exc:
            print(f"Error displaying frame: {exc}")
            self._show_error_on_canvas(f"Frame display error:\\n{exc}")

    def _show_image_on_canvas(self, image: Image.Image):
        """Hiển thị ảnh trên canvas với kích thước responsive"""
        try:
            # Lưu ảnh gốc để resize sau
            self._original_image = image.copy()
            
            # Lấy kích thước canvas thực tế
            self.canvas.update_idletasks()
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            # Nếu canvas chưa có kích thước, dùng giá trị mặc định
            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width = 500
                canvas_height = 400
            
            img_width, img_height = image.size
            if img_width <= 0 or img_height <= 0:
                raise ValueError("Invalid image dimensions.")
            
            # Tính tỷ lệ để fit vào canvas
            ratio = min(canvas_width / img_width, canvas_height / img_height, 1.0)
            new_width = max(1, int(img_width * ratio))
            new_height = max(1, int(img_height * ratio))
            
            resized = image.resize((new_width, new_height), Image.LANCZOS)
            self.img_tk = ImageTk.PhotoImage(resized)
            self.canvas.delete("all")
            # Set background color
            self.canvas.configure(bg="#f1f5f9")
            self.canvas.create_image(canvas_width // 2, canvas_height // 2, image=self.img_tk, anchor="center")
        except Exception as exc:
            print(f"Error showing image on canvas: {exc}")
            self._show_error_on_canvas(f"Canvas display error:\\n{exc}")

    def _show_error_on_canvas(self, message: str):
        self.canvas.delete("all")
        self.canvas.create_text(
            250,
            200,
            text=message,
            font=("Arial", 12),
            fill="red",
            justify="center",
        )

    def _render_video_summary(self, summary_path: Optional[str]):
        if not summary_path or not os.path.exists(summary_path):
            return
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
        except Exception as exc:
            print(f"Error loading video summary: {exc}")
            return
        objects = summary.get("objects", [])
        relations = summary.get("relations", [])

        self.objects_text.delete(1.0, tk.END)
        if objects:
            obj_lines = [f"{entry.get('label','?')}: {entry.get('count',0)}" for entry in objects[:20]]
            self.objects_text.insert(tk.END, "\n".join(obj_lines))
        else:
            self.objects_text.insert(tk.END, "Không có vật thể nào được phát hiện.")

        self.relationships_text.delete(1.0, tk.END)
        if relations:
            rel_lines = []
            for entry in relations[:20]:
                rel_lines.append(
                    f"{entry.get('subject','?')} {entry.get('relation','?')} {entry.get('object','?')} ({entry.get('count',0)})"
                )
            self.relationships_text.insert(tk.END, "\n".join(rel_lines))
        else:
            self.relationships_text.insert(tk.END, "Không có mối quan hệ nào được phát hiện.")

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


    def run_video_demo_thread(self):
        if not self.video_path:
            self.title_label.config(text="❌ Hãy chọn video trước khi chạy demo!")
            return
        if self.video_thread and self.video_thread.is_alive():
            self.title_label.config(text="Video relation demo đang chạy...")
            return
        self.title_label.config(text="Đang chuẩn bị chạy video relation demo...")
        self.video_thread = threading.Thread(target=self.run_video_demo, daemon=True)
        self.video_thread.start()

    def run_video_demo(self):
        try:
            self.video_stop_event.clear()
            if self.video_pipeline is None:
                self.video_pipeline = VideoRelationPipeline(
                    reltr_checkpoint=self.checkpoint_path,
                )

            def handle_frame(frame):
                frame_copy = frame.copy()
                self.root.after(0, lambda f=frame_copy: self.display_frame_from_array(f))

            def handle_relations(payload):
                relations = payload.get("relations", [])
                objects = payload.get("objects", [])
                self.root.after(0, lambda rels=relations: self._update_live_relations(rels))
                self.root.after(0, lambda objs=objects: self._update_live_objects(objs))

            outputs = self.video_pipeline.process_video(
                self.video_path,
                output_dir="video_outputs",
                frame_stride=2,
                on_frame=handle_frame,
                on_relations=handle_relations,
                stop_event=self.video_stop_event,
            )
            self.latest_video_outputs = outputs

            status = (
                "Đã dừng video relation demo."
                if self.video_stop_event.is_set()
                else f"Hoàn tất video demo: {os.path.basename(outputs['video'])}"
            )
            self.root.after(0, lambda msg=status: self.title_label.config(text=msg))
            summary_file = outputs.get("summary")
            self.root.after(0, lambda path=summary_file: self._render_video_summary(path))
        except Exception as exc:
            self.root.after(0, lambda: self.title_label.config(text=f"❌ Lỗi video demo: {exc}"))
            print(f"Video demo error: {exc}")

    def stop_video_demo(self):
        if self.video_thread and self.video_thread.is_alive():
            self.video_stop_event.set()
            self.title_label.config(text="Đang dừng video relation demo...")
        else:
            self.title_label.config(text="Không có video relation demo đang chạy.")

    def _update_live_relations(self, relations):
        self.relationships_text.delete(1.0, tk.END)
        if not relations:
            self.relationships_text.insert(tk.END, "Không có mối quan hệ nào được phát hiện.")
            return
        lines = []
        for rel in relations[:20]:
            subject = rel.get("subject", "unknown")
            relation = rel.get("relation", "liên quan")
            obj = rel.get("object", "unknown")
            confidence = rel.get("confidence", 0.0)
            subj_id = rel.get("subject_track_id")
            obj_id = rel.get("object_track_id")
            prefix = ""
            if subj_id is not None or obj_id is not None:
                prefix = f"[{subj_id or '-'}->{obj_id or '-'}] "
            lines.append(f"{prefix}{subject} {relation} {obj} ({confidence:.2f})")
        self.relationships_text.insert(tk.END, "\n".join(lines))

    def _update_live_objects(self, objects):
        self.objects_text.delete(1.0, tk.END)
        if not objects:
            self.objects_text.insert(tk.END, "Không có vật thể nào được phát hiện.")
            return
        lines = []
        for obj in objects[:20]:
            label = obj.get("class", "object")
            track_id = obj.get("track_id")
            confidence = obj.get("confidence", 0.0)
            if track_id is not None:
                lines.append(f"ID {track_id}: {label} ({confidence:.2f})")
            else:
                lines.append(f"{label} ({confidence:.2f})")
        self.objects_text.insert(tk.END, "\n".join(lines))


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
