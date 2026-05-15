import subprocess
import time
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LAUNCH_FILE = 'start_multi_ai.launch.py'

def print_banner(text):
    print(f"\n{'='*60}")
    print(f" {text}")
    print(f"{'='*60}\n")

def main():
    print_banner("🚀 BẬT HỆ THỐNG AUTO MULTI-MAZE (TỰ ĐỘNG KHÔI PHỤC KHI CRASH)")
    print("Hệ thống sẽ chạy liên tục 48 tiếng. Nếu lỡ văng lỗi hay hết RAM, nó sẽ tự bật lại.")
    
    restart_count = 0
    
    while True:
        try:
            print_banner(f"Bắt đầu chu kỳ huấn luyện (Lần khởi động: {restart_count + 1})")
            
            # Chạy launch file
            result = subprocess.run(['ros2', 'launch', LAUNCH_FILE])
            exit_code = result.returncode
            
        except KeyboardInterrupt:
            print("\n🛑 Nhận tín hiệu dừng từ người dùng! Tắt toàn bộ Auto Multi...")
            break
            
        if exit_code == 130: # Người dùng tự ấn Ctrl+C
            print_banner("🛑 ĐÃ TẮT BẰNG TAY (Ctrl+C). Dừng hệ thống.")
            break
        elif exit_code == 0:
            print_banner("🎉 MÀN HUẤN LUYỆN ĐÃ HOÀN TẤT TRỌN VẸN 1000 THẾ HỆ!")
            break
        else:
            restart_count += 1
            print_banner(f"⚠️ HỆ THỐNG VĂNG LỖI HOẶC TRÀN RAM (Mã lỗi: {exit_code})")
            print("Không sao cả! Dữ liệu đã được Backup an toàn.")
            print("Đang tự động dọn dẹp và khởi động lại sau 5 giây...")
            time.sleep(5)

if __name__ == "__main__":
    main()
