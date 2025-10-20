#!/usr/bin/env python3
"""
Script chạy Reinforcement Learning Training trực tiếp
Dành cho Google Colab và môi trường console
"""

import os
import sys
from app_console import ObjectDetectionConsoleApp

def run_rl_training():
    """Chạy RL Training với cấu hình mặc định"""
    print("🧠 REINFORCEMENT LEARNING TRAINING")
    print("=" * 50)
    
    # Khởi tạo app
    app = ObjectDetectionConsoleApp()
    
    try:
        print("🚀 Bắt đầu RL Training...")
        print("📊 Cấu hình:")
        print("  - Model: YOLOv5 + Relationship Detection")
        print("  - Environment: Object Detection & Relationship Analysis")
        print("  - Algorithm: Custom RL Enhancement")
        print("  - Checkpoint: checkpoint.pth")
        print()
        
        # Chạy RL Training
        results = app.rl_enhancement.run_reinforcement_learning()
        
        print("\n✅ RL TRAINING HOÀN TẤT!")
        print("=" * 30)
        print(f"🎯 Final Reward: {results.get('reward', 0):.4f}")
        print(f"📈 Training Episodes: {results.get('episodes', 0)}")
        print(f"⏱️ Training Time: {results.get('training_time', 0):.2f}s")
        
        # Hiển thị metrics chi tiết nếu có
        if 'metrics' in results:
            metrics = results['metrics']
            print(f"\n📊 CHI TIẾT METRICS:")
            print(f"  - Average Loss: {metrics.get('avg_loss', 0):.4f}")
            print(f"  - Detection Accuracy: {metrics.get('detection_acc', 0):.4f}")
            print(f"  - Relationship Accuracy: {metrics.get('relationship_acc', 0):.4f}")
        
        # Hiển thị files được tạo
        print(f"\n📁 FILES ĐƯỢC TẠO:")
        print("  - rl_training_metrics_*.json")
        print("  - rl_training_summary_*.json") 
        print("  - checkpoint.pth (updated)")
        
        return results
        
    except Exception as e:
        print(f"❌ LỖI RL TRAINING: {e}")
        print("\n🔧 KIỂM TRA:")
        print("  - File checkpoint.pth có tồn tại không?")
        print("  - Có đủ dữ liệu training không?")
        print("  - Có lỗi import module không?")
        return None

def run_rl_with_evaluation():
    """Chạy RL Training và đánh giá kết quả"""
    print("🧠 RL TRAINING + EVALUATION")
    print("=" * 40)
    
    app = ObjectDetectionConsoleApp()
    
    # 1. Chạy RL Training
    print("1️⃣ Chạy RL Training...")
    results = run_rl_training()
    
    if results is None:
        return
    
    # 2. Đánh giá kết quả
    print("\n2️⃣ Đánh giá kết quả training...")
    try:
        app.evaluate_training_results()
    except Exception as e:
        print(f"❌ Lỗi đánh giá: {e}")

def run_synthetic_generation():
    """Chạy tạo dữ liệu synthetic"""
    print("🎨 SYNTHETIC DATA GENERATION")
    print("=" * 40)
    
    app = ObjectDetectionConsoleApp()
    
    try:
        print("🚀 Tạo dữ liệu synthetic...")
        synthetic_data = app.rl_enhancement.generate_synthetic_dataset()
        print(f"✅ Đã tạo {len(synthetic_data)} ảnh synthetic!")
        
        # Hiển thị thông tin chi tiết
        if synthetic_data:
            print(f"\n📊 THÔNG TIN SYNTHETIC DATA:")
            print(f"  - Số lượng ảnh: {len(synthetic_data)}")
            print(f"  - Loại dữ liệu: AI Generated Images")
            print(f"  - Mục đích: Data Augmentation cho RL Training")
        
    except Exception as e:
        print(f"❌ Lỗi tạo synthetic data: {e}")

def main():
    """Main function với menu lựa chọn"""
    print("🤖 REINFORCEMENT LEARNING CONSOLE")
    print("=" * 50)
    print("Chọn chức năng:")
    print("1. Chạy RL Training")
    print("2. Chạy RL Training + Evaluation") 
    print("3. Tạo Synthetic Data")
    print("4. Đánh giá kết quả training")
    print("5. Chạy tất cả (RL + Synthetic + Evaluation)")
    print("6. Thoát")
    
    while True:
        choice = input("\nNhập lựa chọn (1-6): ").strip()
        
        if choice == "1":
            run_rl_training()
            break
            
        elif choice == "2":
            run_rl_with_evaluation()
            break
            
        elif choice == "3":
            run_synthetic_generation()
            break
            
        elif choice == "4":
            app = ObjectDetectionConsoleApp()
            app.evaluate_training_results()
            break
            
        elif choice == "5":
            print("🚀 CHẠY TẤT CẢ CHỨC NĂNG")
            print("=" * 30)
            
            # 1. RL Training
            print("\n1️⃣ RL Training...")
            run_rl_training()
            
            # 2. Synthetic Data
            print("\n2️⃣ Synthetic Data Generation...")
            run_synthetic_generation()
            
            # 3. Evaluation
            print("\n3️⃣ Evaluation...")
            app = ObjectDetectionConsoleApp()
            app.evaluate_training_results()
            
            print("\n✅ HOÀN TẤT TẤT CẢ CHỨC NĂNG!")
            break
            
        elif choice == "6":
            print("👋 Tạm biệt!")
            break
            
        else:
            print("❌ Lựa chọn không hợp lệ!")

if __name__ == "__main__":
    # Kiểm tra arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "train":
            run_rl_training()
        elif sys.argv[1] == "eval":
            run_rl_with_evaluation()
        elif sys.argv[1] == "synthetic":
            run_synthetic_generation()
        else:
            print("❌ Argument không hợp lệ!")
            print("Sử dụng: python run_rl_training.py [train|eval|synthetic]")
    else:
        main()
