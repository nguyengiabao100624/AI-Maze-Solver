#!/usr/bin/env python3
import subprocess
import time
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LAUNCH_FILE = os.path.join(SCRIPT_DIR, 'start_ai.launch.py')

def print_banner(text):
    print(f"\n{'='*60}")
    print(f" 🚀 {text}")
    print(f"{'='*60}\n")

def main():
    map_count = 1
    print_banner("HỆ THỐNG AUTO CURRICULUM ĐÃ KHỞI ĐỘNG!")
    print("Mục tiêu: Đổi map tự động ngay khi bầy xe đủ tiêu chuẩn Tốt nghiệp.")
    
    while True:
        print_banner(f"BẮT ĐẦU TRAINING MÊ CUNG THỨ #{map_count}")
        
        # Gọi launch file
        # Chạy subprocess và in log trực tiếp ra màn hình
        try:
            result = subprocess.run(['ros2', 'launch', LAUNCH_FILE])
            exit_code = result.returncode
        except KeyboardInterrupt:
            print("\n🛑 Nhận tín hiệu dừng từ người dùng! Đang tắt toàn bộ Auto Curriculum...")
            break
            
        # Kiểm tra mã lỗi
        if exit_code == 0:
            print_banner(f"🎉 CHÚC MỪNG! BẦY XE ĐÃ TỐT NGHIỆP MÊ CUNG #{map_count}!")
            print("Đang dọn dẹp bộ nhớ và chuẩn bị sinh Map mới...")
            time.sleep(3) # Nghỉ ngơi cho CPU hạ nhiệt một chút
            map_count += 1
        elif exit_code == 130: # Mã trả về khi ấn Ctrl+C
            print_banner("🛑 ĐÃ TẮT BẰNG TAY (Ctrl+C). Dừng hệ thống.")
            break
        else:
            print_banner(f"⚠️ HỆ THỐNG TẮT BẤT THƯỜNG (Mã lỗi: {exit_code})")
            print("Đang khởi động lại sau 5 giây để tránh kẹt ROS2...")
            time.sleep(5)
            # Khởi động lại vòng lặp ngay cả khi lỗi để đảm bảo chạy 24/7

if __name__ == '__main__':
    main()
