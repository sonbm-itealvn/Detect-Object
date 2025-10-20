#!/usr/bin/env python3
"""
Google Colab Runner for Object Detection & Relationship Analysis
Phiên bản đơn giản để chạy trên Google Colab
"""

import os
import sys
from app_console import ObjectDetectionConsoleApp

def run_on_colab(image_path=None):
    """
    Hàm chính để chạy trên Google Colab
    Args:
        image_path (str): Đường dẫn đến ảnh cần xử lý
    """
    print("🚀 Khởi động Object Detection & Relationship Analysis trên Google Colab")
    print("=" * 70)
    
    # Khởi tạo app
    app = ObjectDetectionConsoleApp()
    
    # Nếu có đường dẫn ảnh được cung cấp
    if image_path:
        if app.set_image_path(image_path):
            print(f"\n🔄 Bắt đầu xử lý ảnh: {image_path}")
            app.run_pipeline()
        else:
            print("❌ Không thể xử lý ảnh được cung cấp")
            return
    else:
        # Chế độ interactive
        print("\n📋 Chế độ Interactive - Chọn chức năng:")
        print("1. Xử lý ảnh mặc định (demo/anh2.jpg)")
        print("2. Chạy RL Training")
        print("3. Tạo dữ liệu synthetic")
        print("4. Đánh giá kết quả training")
        print("5. Tải lại dữ liệu JSON")
        
        choice = input("\nNhập lựa chọn (1-5): ").strip()
        
        if choice == "1":
            # Thử các ảnh demo có sẵn
            demo_images = [
                "demo/anh2.jpg",
                "demo/vg1.jpg", 
                "demo/vg2.jpg",
                "demo/vg3.jpg",
                "demo/vg4.jpg",
                "demo/vg5.jpg"
            ]
            
            for demo_img in demo_images:
                if os.path.exists(demo_img):
                    print(f"✅ Tìm thấy ảnh demo: {demo_img}")
                    if app.set_image_path(demo_img):
                        app.run_pipeline()
                    break
            else:
                print("❌ Không tìm thấy ảnh demo nào")
                
        elif choice == "2":
            app.run_rl_training()
            
        elif choice == "3":
            app.generate_synthetic_data()
            
        elif choice == "4":
            app.evaluate_training_results()
            
        elif choice == "5":
            app.refresh_data()
            
        else:
            print("❌ Lựa chọn không hợp lệ!")

def quick_demo():
    """Chạy demo nhanh với ảnh có sẵn"""
    print("🎯 QUICK DEMO - Xử lý ảnh demo")
    print("=" * 40)
    
    app = ObjectDetectionConsoleApp()
    
    # Tìm ảnh demo
    demo_images = [
        "demo/anh2.jpg",
        "demo/vg1.jpg", 
        "demo/vg2.jpg",
        "demo/vg3.jpg",
        "demo/vg4.jpg",
        "demo/vg5.jpg"
    ]
    
    for demo_img in demo_images:
        if os.path.exists(demo_img):
            print(f"📸 Sử dụng ảnh demo: {demo_img}")
            if app.set_image_path(demo_img):
                app.run_pipeline()
            return
    
    print("❌ Không tìm thấy ảnh demo nào")

if __name__ == "__main__":
    # Kiểm tra nếu chạy trên Google Colab
    try:
        import google.colab
        print("🌐 Đang chạy trên Google Colab")
        IN_COLAB = True
    except ImportError:
        print("💻 Đang chạy trên môi trường local")
        IN_COLAB = False
    
    # Xử lý arguments
    if len(sys.argv) > 1:
        # Chạy với ảnh được chỉ định
        image_path = sys.argv[1]
        run_on_colab(image_path)
    else:
        # Chạy interactive mode
        run_on_colab()
