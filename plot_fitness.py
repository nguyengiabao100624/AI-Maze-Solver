import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import warnings

# Tắt cảnh báo UserWarning của matplotlib về việc thiếu font emoji (Missing Glyph)
warnings.filterwarnings("ignore", category=UserWarning)

def plot_fitness():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(script_dir, 'fitness_log.csv')
    if not os.path.exists(log_file):
        print("Chưa tìm thấy file fitness_log.csv.")
        return
    try:
        df = pd.read_csv(log_file)
        if len(df) < 2:
            print("Chưa đủ dữ liệu (cần ít nhất 2 thế hệ).")
            return

        # Chỉ vẽ tối đa 200 thế hệ gần nhất để tránh rối mắt
        if len(df) > 200:
            df = df.copy().tail(200)

        fig = plt.figure(figsize=(14, 8), facecolor='#1a1a2e')
        gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

        # Màu sắc
        col_bg   = '#1a1a2e'
        col_best = '#00d4aa'
        col_avg  = '#ff6b6b'
        col_mut  = '#ffd93d'

        ax_main = fig.add_subplot(gs[0, :])    # Đồ thị chính (span 2 cột)
        ax_mut  = fig.add_subplot(gs[1, 0])    # Đột biến
        ax_hist = fig.add_subplot(gs[1, 1])    # Histogram điểm cuối

        # ── Đồ thị chính ──────────────────────────────────────────
        ax_main.set_facecolor(col_bg)
        ax_main.plot(df['Generation'], df['Best_Fitness'],
                     label='🏆 Xe Giỏi Nhất', color=col_best,
                     marker='o', markersize=3, linewidth=2)
        ax_main.plot(df['Generation'], df['Avg_Fitness'],
                     label='📊 Điểm Trung Bình', color=col_avg,
                     linestyle='--', linewidth=1.5, alpha=0.8)

        # Vùng tô phủ giữa best và avg
        ax_main.fill_between(df['Generation'], df['Best_Fitness'], df['Avg_Fitness'],
                              alpha=0.1, color=col_best)

        # Đường Moving Average 10 gen
        if len(df) >= 10:
            ma = df['Best_Fitness'].rolling(10).mean()
            ax_main.plot(df['Generation'], ma, color='white',
                         linestyle=':', linewidth=1.5, label='MA-10', alpha=0.6)

        # Đánh dấu kỷ lục
        best_row = df.loc[df['Best_Fitness'].idxmax()]
        ax_main.axhline(y=best_row['Best_Fitness'], color=col_best,
                        linestyle=':', alpha=0.3)
        ax_main.annotate(f"  KỶ LỤC: {best_row['Best_Fitness']:.0f}",
                         xy=(best_row['Generation'], best_row['Best_Fitness']),
                         color=col_best, fontsize=9, va='bottom')

        ax_main.set_xlabel('Thế Hệ', color='white', fontsize=11)
        ax_main.set_ylabel('Điểm Số (Fitness)', color='white', fontsize=11)
        ax_main.set_title('🌌 Tiến Hóa Siêu Cấp Vũ Trụ — Robot v5.0 TURBO (200 Thế Hệ Gần Nhất)',
                          color='white', fontsize=13, fontweight='bold', pad=12)
        ax_main.tick_params(colors='white')
        ax_main.legend(facecolor='#2a2a4e', labelcolor='white', fontsize=9)
        ax_main.grid(True, alpha=0.15, color='white')
        for spine in ax_main.spines.values():
            spine.set_edgecolor('#444')

        # ── Đồ thị Mutation Rate ──────────────────────────────────
        if 'Mutation_Rate' in df.columns:
            ax_mut.set_facecolor(col_bg)
            ax_mut.plot(df['Generation'], df['Mutation_Rate'],
                        color=col_mut, linewidth=2)
            ax_mut.fill_between(df['Generation'], 0, df['Mutation_Rate'],
                                alpha=0.2, color=col_mut)
            ax_mut.set_xlabel('Thế Hệ', color='white')
            ax_mut.set_ylabel('Mutation Rate', color='white')
            ax_mut.set_title('⚡ Tỉ Lệ Đột Biến', color='white', fontsize=10)
            ax_mut.tick_params(colors='white')
            ax_mut.grid(True, alpha=0.15, color='white')
            for spine in ax_mut.spines.values():
                spine.set_edgecolor('#444')

        # ── Histogram điểm số ──────────────────────────────────────
        ax_hist.set_facecolor(col_bg)
        n_last = min(30, len(df))
        last_best = df['Best_Fitness'].tail(n_last).values
        ax_hist.hist(last_best, bins=12, color=col_best, alpha=0.7, edgecolor='white', linewidth=0.5)
        ax_hist.axvline(x=np.median(last_best), color='white',
                        linestyle='--', label=f'Median: {np.median(last_best):.0f}')
        ax_hist.set_xlabel('Điểm Số', color='white')
        ax_hist.set_ylabel('Tần Suất', color='white')
        ax_hist.set_title(f'📈 Phân Bố ({n_last} gen gần nhất)', color='white', fontsize=10)
        ax_hist.tick_params(colors='white')
        ax_hist.legend(facecolor='#2a2a4e', labelcolor='white', fontsize=8)
        ax_hist.grid(True, alpha=0.15, color='white')
        for spine in ax_hist.spines.values():
            spine.set_edgecolor('#444')

        fig.patch.set_facecolor(col_bg)
        chart_file = os.path.join(script_dir, 'fitness_chart.png')
        plt.savefig(chart_file, dpi=150, bbox_inches='tight',
                    facecolor=col_bg)
        print(f"📊 Đã vẽ biểu đồ Siêu Cấp! ({len(df)} thế hệ) → fitness_chart.png")

    except Exception as e:
        print(f"Lỗi: {e}")

if __name__ == "__main__":
    plot_fitness()
