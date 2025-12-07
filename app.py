import json
import subprocess
import threading
import re
import tkinter as tk
from tkinter import Entry, filedialog, Label, Button, Canvas, Frame, Scrollbar, Text, ttk
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
        
        # Ultra Modern Color Scheme - Light & Airy Design
        self.colors = {
            # Backgrounds - Light theme với subtle gradients
            'bg_main': '#f5f7fa',  # Soft gray-blue
            'bg_sidebar': '#ffffff',  # Pure white sidebar
            'bg_card': '#ffffff',
            'bg_card_hover': '#f8fafc',
            'bg_panel': '#fafbfc',
            
            # Accent Colors - Vibrant & Modern
            'accent_primary': '#6366f1',  # Indigo
            'accent_primary_light': '#818cf8',
            'accent_primary_dark': '#4f46e5',
            'accent_success': '#10b981',
            'accent_success_light': '#34d399',
            'accent_warning': '#f59e0b',
            'accent_warning_light': '#fbbf24',
            'accent_danger': '#ef4444',
            'accent_danger_light': '#f87171',
            'accent_purple': '#8b5cf6',
            'accent_purple_light': '#a78bfa',
            'accent_teal': '#14b8a6',
            'accent_teal_light': '#5eead4',
            'accent_orange': '#f97316',
            'accent_orange_light': '#fb923c',
            'accent_cyan': '#06b6d4',
            'accent_cyan_light': '#22d3ee',
            
            # Text Colors
            'text_primary': '#1e293b',
            'text_secondary': '#475569',
            'text_muted': '#94a3b8',
            'text_light': '#ffffff',
            'text_on_accent': '#ffffff',
            
            # Borders & Dividers
            'border': '#e2e8f0',
            'border_light': '#f1f5f9',
            'border_dark': '#cbd5e1',
            
            # Shadows
            'shadow_sm': '#e2e8f0',
            'shadow_md': '#cbd5e1',
            'shadow_lg': '#94a3b8',
            
            # Status Colors
            'status_safe': '#10b981',
            'status_warning': '#f59e0b',
            'status_danger': '#ef4444',
        }
        
        self.root.configure(bg=self.colors['bg_main'])
        
        # Configure ttk style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Bind resize event để responsive
        self.root.bind("<Configure>", self.on_window_resize)

        # ========== MAIN LAYOUT: Sidebar + Content ==========
        main_wrapper = Frame(root, bg=self.colors['bg_main'])
        main_wrapper.pack(fill="both", expand=True)
        
        # ========== LEFT SIDEBAR ==========
        sidebar = Frame(main_wrapper, bg=self.colors['bg_sidebar'], width=280)
        sidebar.pack(side="left", fill="y", padx=(0, 1))
        sidebar.pack_propagate(False)
        
        # Sidebar Header
        sidebar_header = Frame(sidebar, bg=self.colors['accent_primary'], height=120)
        sidebar_header.pack(fill="x")
        sidebar_header.pack_propagate(False)
        
        # Logo/Title in sidebar
        logo_frame = Frame(sidebar_header, bg=self.colors['accent_primary'])
        logo_frame.pack(expand=True, fill="both", padx=20, pady=20)
        
        title_main = Label(logo_frame,
                          text="🔍 AI Vision",
                          font=("Segoe UI", 20, "bold"),
                          bg=self.colors['accent_primary'],
                          fg=self.colors['text_light'])
        title_main.pack(anchor="w", pady=(0, 5))
        
        title_sub = Label(logo_frame,
                         text="Object Detection & Analysis",
                         font=("Segoe UI", 9),
                         bg=self.colors['accent_primary'],
                         fg=self.colors['text_light'])
        title_sub.pack(anchor="w")
        
        # Status indicator
        self.status_indicator = Frame(sidebar_header, bg=self.colors['accent_primary'], height=30)
        self.status_indicator.pack(fill="x", padx=20, pady=(0, 15))
        
        self.title_label = Label(self.status_indicator,
                               text="● Ready",
                               font=("Segoe UI", 10, "bold"),
                               bg=self.colors['accent_primary'],
                               fg=self.colors['text_light'],
                               anchor="w")
        self.title_label.pack(fill="x")
        
        # Sidebar Content - Button Groups
        sidebar_content = Frame(sidebar, bg=self.colors['bg_sidebar'])
        sidebar_content.pack(fill="both", expand=True, padx=15, pady=15)
        
        # Section: Image Operations
        section_label1 = Label(sidebar_content,
                              text="IMAGE OPERATIONS",
                              font=("Segoe UI", 8, "bold"),
                              bg=self.colors['bg_sidebar'],
                              fg=self.colors['text_muted'],
                              anchor="w")
        section_label1.pack(fill="x", pady=(0, 10))
        
        image_ops_frame = Frame(sidebar_content, bg=self.colors['bg_sidebar'])
        image_ops_frame.pack(fill="x", pady=(0, 20))
        
        # Section: Video Operations
        section_label2 = Label(sidebar_content,
                              text="VIDEO OPERATIONS",
                              font=("Segoe UI", 8, "bold"),
                              bg=self.colors['bg_sidebar'],
                              fg=self.colors['text_muted'],
                              anchor="w")
        section_label2.pack(fill="x", pady=(0, 10))
        
        video_ops_frame = Frame(sidebar_content, bg=self.colors['bg_sidebar'])
        video_ops_frame.pack(fill="x", pady=(0, 20))
        
        # Section: Training & Analysis
        section_label3 = Label(sidebar_content,
                              text="TRAINING & ANALYSIS",
                              font=("Segoe UI", 8, "bold"),
                              bg=self.colors['bg_sidebar'],
                              fg=self.colors['text_muted'],
                              anchor="w")
        section_label3.pack(fill="x", pady=(0, 10))
        
        training_ops_frame = Frame(sidebar_content, bg=self.colors['bg_sidebar'])
        training_ops_frame.pack(fill="x", pady=(0, 20))
        
        # Store frames for buttons
        self.image_ops_frame = image_ops_frame
        self.video_ops_frame = video_ops_frame
        self.training_ops_frame = training_ops_frame

        self.rl_enhancement = AppReinforcementLearning(self)
        self.training_evaluator = TrainingEvaluator()

        # Ultra Modern Button Style - Full width, icon + text
        def create_sidebar_button(parent, icon, text, command, color, hover_color=None):
            if hover_color is None:
                hover_color = color
            
            btn_container = Frame(parent, bg=self.colors['bg_sidebar'], relief="flat", bd=0)
            btn_container.pack(fill="x", pady=4)
            
            btn = Button(btn_container,
                        text=f"{icon}  {text}",
                        command=command,
                        font=("Segoe UI", 10),
                        bg=self.colors['bg_sidebar'],
                        fg=self.colors['text_primary'],
                        relief="flat",
                        bd=0,
                        anchor="w",
                        padx=15,
                        pady=12,
                        cursor="hand2",
                        activebackground=self.colors['bg_card_hover'],
                        activeforeground=self.colors['text_primary'])
            btn.pack(fill="x")
            
            # Hover effect với border highlight
            def on_enter(e):
                btn.configure(bg=self.colors['bg_card_hover'], fg=color)
                btn_container.configure(bg=self.colors['bg_card_hover'])
            def on_leave(e):
                btn.configure(bg=self.colors['bg_sidebar'], fg=self.colors['text_primary'])
                btn_container.configure(bg=self.colors['bg_sidebar'])
            
            btn.bind("<Enter>", on_enter)
            btn.bind("<Leave>", on_leave)
            btn_container.bind("<Enter>", on_enter)
            btn_container.bind("<Leave>", on_leave)
            
            return btn

        # Image Operations Buttons
        self.btn_select = create_sidebar_button(
            self.image_ops_frame, "📁", "Select Image", self.select_image,
            self.colors['accent_primary']
        )
        self.btn_run = create_sidebar_button(
            self.image_ops_frame, "▶️", "Detect Objects", self.run_pipeline_thread,
            self.colors['accent_success']
        )
        self.btn_refresh = create_sidebar_button(
            self.image_ops_frame, "🔄", "Reload Data", self.refresh_data,
            self.colors['accent_warning']
        )
        
        # Video Operations Buttons
        self.btn_select_video = create_sidebar_button(
            self.video_ops_frame, "📹", "Select Video", self.select_video,
            self.colors['accent_teal']
        )
        self.btn_run_video = create_sidebar_button(
            self.video_ops_frame, "▶️", "Run Video Demo", self.run_video_demo_thread,
            self.colors['accent_cyan']
        )
        self.btn_stop_video = create_sidebar_button(
            self.video_ops_frame, "⏹️", "Stop Video", self.stop_video_demo,
            self.colors['accent_danger']
        )
        
        # Training & Analysis Buttons
        self.btn_rl_train = create_sidebar_button(
            self.training_ops_frame, "🧠", "RL Training", self.run_rl_training,
            self.colors['accent_purple']
        )
        self.btn_generate_synthetic = create_sidebar_button(
            self.training_ops_frame, "🎨", "Generate Synthetic", self.generate_synthetic_data,
            self.colors['accent_orange']
        )
        self.btn_evaluate_training = create_sidebar_button(
            self.training_ops_frame, "📊", "Evaluate Training", self.evaluate_training_results,
            self.colors['accent_purple']
        )

        # ========== MAIN CONTENT AREA ==========
        content_area = Frame(main_wrapper, bg=self.colors['bg_main'])
        content_area.pack(side="left", fill="both", expand=True)
        
        # Top Status Bar
        status_bar = Frame(content_area, bg=self.colors['bg_card'], height=60)
        status_bar.pack(fill="x", padx=15, pady=(15, 0))
        status_bar.pack_propagate(False)
        
        status_inner = Frame(status_bar, bg=self.colors['bg_card'])
        status_inner.pack(fill="both", expand=True, padx=20, pady=15)
        
        self.alert_normal_bg = self.colors['status_safe']
        self.alert_normal_fg = self.colors['text_light']
        self.alert_warning_bg = self.colors['status_danger']
        self.alert_warning_fg = self.colors['text_light']
        
        self.alert_label = Label(
            status_inner,
            text="🟢 Safe Zone: Ready (>=2m)",
            font=("Segoe UI", 11, "bold"),
            bg=self.alert_normal_bg,
            fg=self.alert_normal_fg,
            padx=20,
            pady=8,
            relief="flat",
            bd=0
        )
        self.alert_label.pack(side="left")

        # ========== MAIN CONTENT GRID ==========
        main_container = Frame(content_area, bg=self.colors['bg_main'])
        main_container.pack(fill="both", expand=True, padx=15, pady=15)
        
        # Grid layout - 2 columns: Image (left) + Info Panels (right)
        main_container.grid_columnconfigure(0, weight=2, minsize=500)
        main_container.grid_columnconfigure(1, weight=1, minsize=350)
        main_container.grid_rowconfigure(0, weight=1)
        
        # ========== LEFT: Image Display Card ==========
        image_card_container = Frame(main_container, bg=self.colors['bg_main'])
        image_card_container.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        # Card với shadow effect (simulated với border)
        image_card = Frame(image_card_container, bg=self.colors['bg_card'], relief="flat", bd=1, highlightbackground=self.colors['border'])
        image_card.pack(fill="both", expand=True)
        
        # Card Header
        image_header = Frame(image_card, bg=self.colors['bg_card'], height=60)
        image_header.pack(fill="x")
        image_header.pack_propagate(False)
        
        image_header_inner = Frame(image_header, bg=self.colors['bg_card'])
        image_header_inner.pack(fill="both", expand=True, padx=20, pady=15)
        
        image_title = Label(image_header_inner,
                           text="🖼️ Image Preview",
                           font=("Segoe UI", 13, "bold"),
                           bg=self.colors['bg_card'],
                           fg=self.colors['text_primary'],
                           anchor="w")
        image_title.pack(side="left")
        
        # Canvas container
        canvas_container = Frame(image_card, bg=self.colors['bg_panel'])
        canvas_container.pack(fill="both", expand=True, padx=20, pady=20)
        
        self.canvas = Canvas(canvas_container,
                           bg=self.colors['bg_panel'],
                           relief="flat",
                           highlightthickness=1,
                           highlightbackground=self.colors['border_light'],
                           highlightcolor=self.colors['accent_primary'])
        self.canvas.pack(fill="both", expand=True)
        
        # Empty state với styling tốt hơn
        self.canvas_empty_text = self.canvas.create_text(
            300, 250,
            text="📷 No Image Selected\n\nClick 'Select Image' in the sidebar\nto get started",
            font=("Segoe UI", 13),
            fill=self.colors['text_muted'],
            justify="center",
            tags="empty_state"
        )
        
        # ========== RIGHT: Info Panels ==========
        right_panel = Frame(main_container, bg=self.colors['bg_main'])
        right_panel.grid(row=0, column=1, sticky="nsew")
        right_panel.grid_rowconfigure(0, weight=1)
        right_panel.grid_rowconfigure(1, weight=1)
        right_panel.grid_columnconfigure(0, weight=1)
        
        # Panel 1: Detected Objects
        objects_card = Frame(right_panel, bg=self.colors['bg_card'], relief="flat", bd=1, highlightbackground=self.colors['border'])
        objects_card.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
        
        objects_header = Frame(objects_card, bg=self.colors['bg_card'], height=50)
        objects_header.pack(fill="x")
        objects_header.pack_propagate(False)
        
        objects_header_inner = Frame(objects_header, bg=self.colors['bg_card'])
        objects_header_inner.pack(fill="both", expand=True, padx=15, pady=12)
        
        objects_title = Label(objects_header_inner,
                            text="📦 Detected Objects",
                            font=("Segoe UI", 12, "bold"),
                            bg=self.colors['bg_card'],
                            fg=self.colors['text_primary'],
                            anchor="w")
        objects_title.pack(side="left")
        
        objects_content = Frame(objects_card, bg=self.colors['bg_card'])
        objects_content.pack(fill="both", expand=True, padx=15, pady=15)
        
        self.objects_text = Text(objects_content,
                               font=("Segoe UI", 9),
                               bg=self.colors['bg_panel'],
                               fg=self.colors['text_primary'],
                               relief="flat",
                               bd=0,
                               wrap="word",
                               padx=12,
                               pady=12,
                               selectbackground=self.colors['accent_success'],
                               selectforeground="white")
        objects_scrollbar = Scrollbar(objects_content,
                                     orient="vertical",
                                     command=self.objects_text.yview,
                                     bg=self.colors['bg_card'],
                                     troughcolor=self.colors['bg_panel'],
                                     activebackground=self.colors['accent_success'],
                                     width=10)
        self.objects_text.configure(yscrollcommand=objects_scrollbar.set)
        
        self.objects_text.pack(side="left", fill="both", expand=True)
        objects_scrollbar.pack(side="right", fill="y")
        
        self.objects_text.insert("1.0", "📋 Objects will appear here\n\nRun 'Detect Objects' to see results")
        self.objects_text.config(state="disabled")
        
        # Panel 2: Relationships
        relationships_card = Frame(right_panel, bg=self.colors['bg_card'], relief="flat", bd=1, highlightbackground=self.colors['border'])
        relationships_card.grid(row=1, column=0, sticky="nsew")
        
        relationships_header = Frame(relationships_card, bg=self.colors['bg_card'], height=50)
        relationships_header.pack(fill="x")
        relationships_header.pack_propagate(False)
        
        relationships_header_inner = Frame(relationships_header, bg=self.colors['bg_card'])
        relationships_header_inner.pack(fill="both", expand=True, padx=15, pady=12)
        
        relationships_title = Label(relationships_header_inner,
                                  text="🔗 Relationships",
                                  font=("Segoe UI", 12, "bold"),
                                  bg=self.colors['bg_card'],
                                  fg=self.colors['text_primary'],
                                  anchor="w")
        relationships_title.pack(side="left")
        
        relationships_content = Frame(relationships_card, bg=self.colors['bg_card'])
        relationships_content.pack(fill="both", expand=True, padx=15, pady=15)
        
        self.relationships_text = Text(relationships_content,
                                     font=("Segoe UI", 9),
                                     bg=self.colors['bg_panel'],
                                     fg=self.colors['text_primary'],
                                     relief="flat",
                                     bd=0,
                                     wrap="word",
                                     padx=12,
                                     pady=12,
                                     selectbackground=self.colors['accent_purple'],
                                     selectforeground="white")
        relationships_scrollbar = Scrollbar(relationships_content,
                                         orient="vertical",
                                         command=self.relationships_text.yview,
                                         bg=self.colors['bg_card'],
                                         troughcolor=self.colors['bg_panel'],
                                         activebackground=self.colors['accent_purple'],
                                         width=10)
        self.relationships_text.configure(yscrollcommand=relationships_scrollbar.set)
        
        self.relationships_text.pack(side="left", fill="both", expand=True)
        relationships_scrollbar.pack(side="right", fill="y")
        
        self.relationships_text.insert("1.0", "🔗 Relationships will appear here\n\nRun 'Detect Objects' to see results")
        self.relationships_text.config(state="disabled")
        
        # Store references
        self.main_container = main_container
        self.image_frame = image_card
        self.objects_frame = objects_card
        self.relationships_frame = relationships_card

        self.image_path = None
        self.result_image_path = "result.jpg"
        self.result_json_path = "converted_bboxes.json"
        self.relationship_json_path = "relationships.json"
        self.checkpoint_path = "reltr_finetuned.pth"

        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.video_path = None
        self.video_pipeline: Optional[VideoRelationPipeline] = None
        self.video_thread: Optional[threading.Thread] = None
        self.video_stop_event = threading.Event()
        self.latest_video_outputs = {}
        self.safe_zone_radius_m = 2.0

    def on_window_resize(self, event=None):
        """Xử lý khi window resize để responsive"""
        if event and event.widget == self.root:
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
        # Xóa empty state nếu có
        self.canvas.delete("empty_state")
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
            self.canvas.configure(bg="#f8fafc")
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

        self.objects_text.config(state="normal")
        self.objects_text.delete(1.0, tk.END)
        if objects:
            header = f"📊 Video Summary - Total: {len(objects)} object types\n{'='*30}\n\n"
            self.objects_text.insert(tk.END, header)
            for i, entry in enumerate(objects[:20], 1):
                label = entry.get('label', '?')
                count = entry.get('count', 0)
                self.objects_text.insert(tk.END, f"🔸 {i}. {label.upper()}: {count} occurrences\n")
        else:
            self.objects_text.insert(tk.END, "❌ No objects detected in video.")
        self.objects_text.config(state="disabled")

        self.relationships_text.config(state="normal")
        self.relationships_text.delete(1.0, tk.END)
        if relations:
            header = f"🔗 Video Summary - Total: {len(relations)} relationships\n{'='*30}\n\n"
            self.relationships_text.insert(tk.END, header)
            for i, entry in enumerate(relations[:20], 1):
                subject = entry.get('subject', '?')
                relation = entry.get('relation', '?')
                obj = entry.get('object', '?')
                count = entry.get('count', 0)
                self.relationships_text.insert(tk.END, f"🔸 {i}. {subject.upper()}\n")
                self.relationships_text.insert(tk.END, f"   🔗 {relation.upper()}\n")
                self.relationships_text.insert(tk.END, f"   🎯 {obj.upper()}\n")
                self.relationships_text.insert(tk.END, f"   📊 Count: {count}\n\n")
        else:
            self.relationships_text.insert(tk.END, "❌ No relationships detected in video.")
        self.relationships_text.config(state="disabled")

    def load_and_display_objects(self):
        """Tải và hiển thị danh sách vật thể từ JSON"""
        try:
            self.objects_text.config(state="normal")
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
                self.objects_text.insert(tk.END, "❌ No objects detected\n\nPlease run 'Detect Objects' to detect objects in the image.")
                self.objects_text.config(state="disabled")
                return

            # Xóa nội dung cũ
            self.objects_text.delete(1.0, tk.END)
            
            # Hiển thị thông tin vật thể với formatting đẹp hơn
            header = f"📊 Total: {len(objects)} objects\n{'='*30}\n\n"
            self.objects_text.insert(tk.END, header)
            
            for i, obj in enumerate(objects, 1):
                class_name = obj.get("class", "Unknown")
                bbox = obj.get("bbox", [])
                confidence = obj.get("confidence", 0)
                
                if len(bbox) >= 4:
                    x, y, w, h = bbox[:4]
                    info = f"🔸 {i}. {class_name.upper()}\n"
                    info += f"   📍 Position: ({int(x)}, {int(y)})\n"
                    info += f"   📏 Size: {int(w-x)} x {int(h-y)} px\n"
                    if isinstance(confidence, (int, float)):
                        info += f"   🎯 Confidence: {confidence:.2%}\n\n"
                    else:
                        info += f"   🎯 Confidence: {confidence}\n\n"
                else:
                    info = f"🔸 {i}. {class_name.upper()}\n"
                    info += f"   ⚠️ Invalid bbox information\n\n"
                
                self.objects_text.insert(tk.END, info)
            
            self.objects_text.config(state="disabled")
                
        except FileNotFoundError:
            self.objects_text.config(state="normal")
            self.objects_text.delete(1.0, tk.END)
            self.objects_text.insert(tk.END, f"❌ File not found: {self.result_json_path}\n\nPlease run 'Detect Objects' first.")
            self.objects_text.config(state="disabled")
        except json.JSONDecodeError:
            self.objects_text.config(state="normal")
            self.objects_text.delete(1.0, tk.END)
            self.objects_text.insert(tk.END, "❌ Error reading JSON file\n\nFile may be corrupted or invalid format.")
            self.objects_text.config(state="disabled")
        except Exception as e:
            self.objects_text.config(state="normal")
            self.objects_text.delete(1.0, tk.END)
            self.objects_text.insert(tk.END, f"❌ Error: {str(e)}")
            self.objects_text.config(state="disabled")

    def load_and_display_relationships(self):
        """Tải và hiển thị danh sách mối quan hệ từ JSON"""
        try:
            self.relationships_text.config(state="normal")
            with open(self.relationship_json_path, "r", encoding="utf-8") as f:
                relationships = json.load(f)
            
            if not relationships:
                self.relationships_text.delete(1.0, tk.END)
                self.relationships_text.insert(tk.END, "❌ No relationships detected\n\nPlease run 'Detect Objects' to analyze relationships.")
                self.relationships_text.config(state="disabled")
                return

            # Xóa nội dung cũ
            self.relationships_text.delete(1.0, tk.END)
            
            # Header
            header = f"🔗 Total: {len(relationships)} relationships\n{'='*30}\n\n"
            self.relationships_text.insert(tk.END, header)
            
            # Hiển thị thông tin mối quan hệ với formatting đẹp hơn
            for i, rel in enumerate(relationships, 1):
                subject = rel.get("subject", "Unknown")
                relation = rel.get("relation", "Unknown")
                obj = rel.get("object", "Unknown")
                similarity = rel.get("visual_similarity", 0)
                confidence = rel.get("confidence", 0)
                
                # Màu sắc dựa trên độ tin cậy (nếu có visual_similarity)
                if similarity > 0:
                    confidence_color = "🟢" if similarity > 0.7 else "🟡" if similarity > 0.4 else "🔴"
                    info = f"{confidence_color} {i}. {subject.upper()}\n"
                    info += f"   🔗 {relation.upper()}\n"
                    info += f"   🎯 {obj.upper()}\n"
                    info += f"   📊 Confidence: {similarity:.2%}\n\n"
                elif confidence > 0:
                    info = f"🔸 {i}. {subject.upper()}\n"
                    info += f"   🔗 {relation.upper()}\n"
                    info += f"   🎯 {obj.upper()}\n"
                    info += f"   📊 Confidence: {confidence:.2%}\n\n"
                else:
                    # Nếu không có visual_similarity, hiển thị đơn giản
                    info = f"🔸 {i}. {subject.upper()}\n"
                    info += f"   🔗 {relation.upper()}\n"
                    info += f"   🎯 {obj.upper()}\n\n"
                
                self.relationships_text.insert(tk.END, info)
            
            self.relationships_text.config(state="disabled")
                
        except FileNotFoundError:
            self.relationships_text.config(state="normal")
            self.relationships_text.delete(1.0, tk.END)
            self.relationships_text.insert(tk.END, f"❌ File not found: {self.relationship_json_path}\n\nPlease run 'Detect Objects' first.")
            self.relationships_text.config(state="disabled")
        except json.JSONDecodeError:
            self.relationships_text.config(state="normal")
            self.relationships_text.delete(1.0, tk.END)
            self.relationships_text.insert(tk.END, "❌ Error reading JSON file\n\nFile may be corrupted or invalid format.")
            self.relationships_text.config(state="disabled")
        except Exception as e:
            self.relationships_text.config(state="normal")
            self.relationships_text.delete(1.0, tk.END)
            self.relationships_text.insert(tk.END, f"❌ Error: {str(e)}")
            self.relationships_text.config(state="disabled")

    def refresh_data(self):
        """Tải lại dữ liệu JSON mà không cần chạy lại pipeline"""
        self.title_label.config(text="🔄 Reloading data...")
        self.load_and_display_objects()
        self.load_and_display_relationships()
        self.title_label.config(text="✅ Data reloaded successfully!")

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
                print("❌ No data to draw bbox")
                return
            
            # Tìm ảnh kết quả
            image_dir = os.path.dirname(self.image_path)
            image_id = os.path.splitext(os.path.basename(self.image_path))[0]
            output_images = glob.glob(f"**/output_{image_id}.jpg", recursive=True)
            
            if not output_images:
                print("❌ Result image not found to draw bbox")
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
            print(f"✅ Relationship bbox drawn and saved at: {result_path}")
            
        except Exception as e:
            print(f"❌ Error drawing relationship bbox: {e}")
            self.title_label.config(text=f"❌ Error drawing bbox: {e}")

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
                self.label.config(text=f"❌ Subject not found: {subject_name} in JSON!")
                print("❌ Error finding subject:", subject_name)
                return

            if object_name and not object_box:
                self.label.config(text=f"❌ Object not found: {object_name} in JSON!")
                print("❌ Error finding object:", object_name)
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

            self.label.config(text="✅ Box drawn successfully!")

        except Exception as e:
            self.label.config(text=f"❌ Error drawing box: {e}")
            print(f"❌ Error drawing box: {e}")

    def run_pipeline_thread(self):
        thread = threading.Thread(target=self.run_pipeline)
        thread.start()

    def run_pipeline(self):
        if not self.image_path:
            self.title_label.config(text="❌ Please select an image first!")
            return

        self.title_label.config(text="⏳ Processing... Please wait.")

        try:
            # 1️⃣ Chạy detect_objects.py
            self.title_label.config(text="🔍 Detecting objects...")
            detect_thread = threading.Thread(target=subprocess.run, args=(["python", "detect_objects.py", self.image_path],))
            detect_thread.start()
            detect_thread.join()  # Đợi detect_objects.py chạy xong

            # 2️⃣ Chạy convert_yolo_to_reltr.py (sau khi detect_objects.py hoàn tất)
            self.title_label.config(text="🔄 Converting YOLO data...")
            convert_thread = threading.Thread(target=subprocess.run, args=(["python", "convert_yolo_to_reltr.py", "result.json"],))
            convert_thread.start()
            convert_thread.join()  # Đợi convert_yolo_to_reltr.py chạy xong

            # 3️⃣ Chạy boundingbox_objects.py (sau khi convert_yolo_to_reltr.py hoàn tất)
            self.title_label.config(text="🔗 Analyzing relationships between objects...")
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
                self.title_label.config(text="✅ Complete! Here are the results.")
            else:
                print("📂 File list in directory:", os.listdir(image_dir))  # Debug kiểm tra
                self.title_label.config(text="❌ Result image not found!")

            # 4️⃣ Tải và hiển thị dữ liệu JSON
            self.title_label.config(text="📊 Loading result data...")
            self.load_and_display_objects()
            self.load_and_display_relationships()
            
            # 5️⃣ Vẽ bbox mối quan hệ trên ảnh
            self.title_label.config(text="🎨 Drawing relationship bboxes...")
            self.draw_relationship_boxes_on_image()
            self.title_label.config(text="✅ Complete! Data loaded and bboxes drawn.")
            
        except Exception as e:
            self.title_label.config(text=f"❌ Error: {e}")
            print(f"❌ Error occurred: {e}")


    def run_video_demo_thread(self):
        if not self.video_path:
            self.title_label.config(text="❌ Please select a video before running demo!")
            return
        if self.video_thread and self.video_thread.is_alive():
            self.title_label.config(text="Video relation demo is running...")
            return
        self.title_label.config(text="Preparing to run video relation demo...")
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
                intrusions = payload.get("intrusions", [])
                danger = payload.get("danger", False)
                self.root.after(0, lambda rels=relations, intr=intrusions, dan=danger: self._update_live_relations(rels, intr, dan))
                self.root.after(0, lambda objs=objects: self._update_live_objects(objs))
                self.root.after(0, lambda intr=intrusions, dan=danger: self._update_alert_banner(intr, dan))

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
                "Video relation demo stopped."
                if self.video_stop_event.is_set()
                else f"Video demo completed: {os.path.basename(outputs['video'])}"
            )
            self.root.after(0, lambda msg=status: self.title_label.config(text=msg))
            summary_file = outputs.get("summary")
            self.root.after(0, lambda path=summary_file: self._render_video_summary(path))
        except Exception as exc:
            self.root.after(0, lambda: self.title_label.config(text=f"❌ Video demo error: {exc}"))
            print(f"Video demo error: {exc}")

    def stop_video_demo(self):
        if self.video_thread and self.video_thread.is_alive():
            self.video_stop_event.set()
            self.title_label.config(text="Stopping video relation demo...")
        else:
            self.title_label.config(text="No video relation demo is running.")

    def _update_live_relations(self, relations, intrusions=None, danger=False):
        self.relationships_text.config(state="normal")
        self.relationships_text.delete(1.0, tk.END)
        lines = []
        intrusions = intrusions or []
        if danger and intrusions:
            lines.append("⚠️ WARNING: safety zone (2m) intrusions ⚠️")
            lines.append("="*30)
            for alert in intrusions[:5]:
                label = alert.get("class", "object")
                track_id = alert.get("track_id")
                dist = alert.get("distance_m")
                descriptor = f"{label}"
                if track_id is not None:
                    descriptor += f" #{track_id}"
                if dist is not None:
                    descriptor += f" @ {dist:.1f} m"
                lines.append(f"  • {descriptor}")
            lines.append("")
        if not relations:
            lines.append("❌ No relationships detected.")
        else:
            header = f"🔗 Total: {len(relations)} relationships\n{'='*30}\n\n"
            lines.append(header)
            for rel in relations[:20]:
                subject = rel.get("subject", "unknown")
                relation = rel.get("relation", "liên quan")
                obj = rel.get("object", "unknown")
                confidence = rel.get("confidence", 0.0)
                subj_id = rel.get("subject_track_id")
                obj_id = rel.get("object_track_id")
                prefix = ""
                if subj_id is not None or obj_id is not None:
                    prefix = f"[{subj_id or '-'}→{obj_id or '-'}] "
                lines.append(f"🔸 {prefix}{subject.upper()}")
                lines.append(f"   🔗 {relation.upper()}")
                lines.append(f"   🎯 {obj.upper()}")
                lines.append(f"   📊 Confidence: {confidence:.2%}\n")
        self.relationships_text.insert(tk.END, '\n'.join(lines))
        self.relationships_text.config(state="disabled")

    def _update_live_objects(self, objects):
        self.objects_text.config(state="normal")
        self.objects_text.delete(1.0, tk.END)
        if not objects:
            self.objects_text.insert(tk.END, "❌ No objects detected.")
            self.objects_text.config(state="disabled")
            return
        lines = []
        header = f"📊 Total: {len(objects)} objects\n{'='*30}\n\n"
        lines.append(header)
        for i, obj in enumerate(objects[:20], 1):
            label = obj.get("class", "object")
            track_id = obj.get("track_id")
            confidence = obj.get("confidence", 0.0)
            if track_id is not None:
                lines.append(f"🔸 {i}. {label.upper()} (ID: {track_id})")
            else:
                lines.append(f"🔸 {i}. {label.upper()}")
            lines.append(f"   🎯 Confidence: {confidence:.2%}\n")
        self.objects_text.insert(tk.END, "\n".join(lines))
        self.objects_text.config(state="disabled")
        self.objects_text.config(state="disabled")


    def _update_alert_banner(self, intrusions, danger):
        intrusions = intrusions or []
        if danger and intrusions:
            nearest = min(
                (alert.get('distance_m') if alert.get('distance_m') is not None else self.safe_zone_radius_m)
                for alert in intrusions
            )
            labels = []
            for alert in intrusions[:3]:
                label = alert.get('class', 'object')
                track_id = alert.get('track_id')
                if track_id is not None:
                    labels.append(f"{label}#{track_id}")
                else:
                    labels.append(label)
            alert_text = (
                f"⚠️ WARNING: {len(intrusions)} object(s) < {self.safe_zone_radius_m:.1f}m "
                f"({', '.join(labels)}) | Nearest: {nearest:.1f}m"
            )
            self.alert_label.config(text=alert_text, bg=self.alert_warning_bg, fg=self.alert_warning_fg)
        else:
            self.alert_label.config(
                text=f"🟢 Safe Zone: Clear (>{self.safe_zone_radius_m:.1f}m)",
                bg=self.alert_normal_bg,
                fg=self.alert_normal_fg
            )

    def run_rl_training(self):
        """Run reinforcement learning training in separate thread"""
        self.title_label.config(text="🧠 Running RL Training...")
        
        def rl_training_thread():
            try:
                results = self.rl_enhancement.run_reinforcement_learning()
                self.title_label.config(text=f"✅ RL Training completed! Reward: {results['reward']:.3f}")
            except Exception as e:
                self.title_label.config(text=f"❌ RL Training error: {e}")
                print(f"❌ RL Training error: {e}")
        
        # Chạy RL training trong thread riêng để không block UI
        thread = threading.Thread(target=rl_training_thread)
        thread.daemon = True  # Thread sẽ tự động kết thúc khi app đóng
        thread.start()
    
    def generate_synthetic_data(self):
        """Generate synthetic dataset from relationships in separate thread"""
        self.title_label.config(text="🎨 Generating synthetic data...")
        
        def synthetic_generation_thread():
            try:
                synthetic_data = self.rl_enhancement.generate_synthetic_dataset()
                self.title_label.config(text=f"✅ Generated {len(synthetic_data)} synthetic images!")
            except Exception as e:
                self.title_label.config(text=f"❌ Synthetic data generation error: {e}")
                print(f"❌ Synthetic data generation error: {e}")
        
        # Chạy synthetic data generation trong thread riêng
        thread = threading.Thread(target=synthetic_generation_thread)
        thread.daemon = True
        thread.start()
    
    def evaluate_training_results(self):
        """Evaluate training results and create comprehensive report"""
        self.title_label.config(text="📊 Evaluating training results...")
        
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
                    
                    self.title_label.config(text=f"✅ Evaluation completed! {total_sessions} sessions, Reward: {avg_reward:.3f}, AI Images: {total_images}")
                    
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
                    self.title_label.config(text="❌ No training results found to evaluate")
                    
            except Exception as e:
                self.title_label.config(text=f"❌ Evaluation error: {e}")
                print(f"❌ Evaluation error: {e}")
        
        # Chạy evaluation trong thread riêng
        thread = threading.Thread(target=evaluation_thread)
        thread.daemon = True
        thread.start()

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()
