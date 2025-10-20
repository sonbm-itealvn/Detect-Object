#!/usr/bin/env python3
"""
Script quản lý experiments cho Reinforcement Learning
Cho phép xem, so sánh và export các experiments
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from RL.experiment_viewer import ExperimentViewer

def main():
    """Main function"""
    print("🔬 EXPERIMENT MANAGER")
    print("=" * 50)
    print("Quản lý và xem lại các experiments Reinforcement Learning")
    print()
    
    viewer = ExperimentViewer()
    
    # Kiểm tra xem có experiments không
    experiments = viewer.list_all_experiments()
    
    if not experiments:
        print("❌ Không có experiment nào được tìm thấy")
        print("💡 Hãy chạy RL Training trước để tạo experiments")
        return
    
    # Menu tương tác
    viewer.interactive_menu()

if __name__ == "__main__":
    main()
