#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_trajectory.py
==================
離線資料視覺化腳本：讀取 odom_recorder_node 產出的 CSV，
繪製 2D 軌跡比較圖，並計算 RMSE。

依賴套件：
  pip3 install pandas matplotlib

執行方式：
  python3 plot_trajectory.py                          # 預設讀取 ~/trajectory_data.csv
  python3 plot_trajectory.py /path/to/custom.csv      # 指定路徑
"""

import sys
import os
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ────────────────────────────────────────────────────────────
#  工具函式
# ────────────────────────────────────────────────────────────
def angle_wrap(angle: float) -> float:
    """將角度歸一化到 (-π, π]。"""
    return math.atan2(math.sin(angle), math.cos(angle))


def compute_translation_rmse(x1, y1, x2, y2) -> float:
    """計算平移 RMSE（歐幾里得距離）。"""
    dx = np.asarray(x1) - np.asarray(x2)
    dy = np.asarray(y1) - np.asarray(y2)
    return float(np.sqrt(np.mean(dx**2 + dy**2)))


def compute_rotation_rmse_deg(yaw1, yaw2) -> float:
    """計算旋轉 RMSE，輸出單位為度 (°)。"""
    diff = np.array([angle_wrap(a - b) for a, b in zip(yaw1, yaw2)])
    rmse_rad = float(np.sqrt(np.mean(diff**2)))
    return math.degrees(rmse_rad)


# ────────────────────────────────────────────────────────────
#  主函式
# ────────────────────────────────────────────────────────────
def main():
    # ---- 解析檔案路徑 ----
    if len(sys.argv) >= 2:
        csv_path = sys.argv[1]
    else:
        csv_path = os.path.expanduser('~/trajectory_data.csv')

    if not os.path.isfile(csv_path):
        print(f'[ERROR] 找不到 CSV 檔案: {csv_path}')
        sys.exit(1)

    # ---- 讀取 CSV ----
    df = pd.read_csv(csv_path)
    required_cols = [
        'raw_x', 'raw_y', 'raw_yaw',
        'ekf_x', 'ekf_y', 'ekf_yaw',
        'gt_x',  'gt_y',  'gt_yaw',
    ]
    for col in required_cols:
        if col not in df.columns:
            print(f'[ERROR] CSV 缺少欄位: {col}')
            sys.exit(1)

    # 丟棄含 NaN 的行（例如 TF 尚未就緒時）
    df = df.dropna(subset=required_cols).reset_index(drop=True)
    if df.empty:
        print('[ERROR] CSV 無有效資料（全為 NaN）。')
        sys.exit(1)

    print(f'[INFO] 讀取 {len(df)} 筆有效資料 from {csv_path}')

    # ---- 計算 RMSE ----
    raw_trans_rmse = compute_translation_rmse(
        df['raw_x'], df['raw_y'], df['gt_x'], df['gt_y'])
    ekf_trans_rmse = compute_translation_rmse(
        df['ekf_x'], df['ekf_y'], df['gt_x'], df['gt_y'])

    raw_rot_rmse = compute_rotation_rmse_deg(df['raw_yaw'], df['gt_yaw'])
    ekf_rot_rmse = compute_rotation_rmse_deg(df['ekf_yaw'], df['gt_yaw'])

    # ---- 終端機輸出 ----
    print('=' * 55)
    print(f'{"":20s} {"平移 RMSE (m)":>15s}  {"旋轉 RMSE (°)":>15s}')
    print('-' * 55)
    print(f'{"Raw Odom vs GT":20s} {raw_trans_rmse:>15.4f}  {raw_rot_rmse:>15.4f}')
    print(f'{"EKF Odom vs GT":20s} {ekf_trans_rmse:>15.4f}  {ekf_rot_rmse:>15.4f}')
    print('=' * 55)

    # ---- 繪製 2D 軌跡 ----
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(df['gt_x'],  df['gt_y'],  color='black', linewidth=2.0,
            label='Ground Truth (SLAM)')
    ax.plot(df['raw_x'], df['raw_y'], color='red',   linewidth=1.2,
            linestyle='--', label='Raw Wheel Odom')
    ax.plot(df['ekf_x'], df['ekf_y'], color='blue',  linewidth=1.2,
            linestyle='-.', label='EKF Fused Odom')

    # 標記起點
    ax.plot(df['gt_x'].iloc[0], df['gt_y'].iloc[0],
            marker='o', color='green', markersize=10, zorder=5, label='Start')

    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('2D Trajectory Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.3)

    # RMSE 文字標註
    rmse_text = (
        f'Raw  Odom → Trans RMSE: {raw_trans_rmse:.4f} m,  '
        f'Rot RMSE: {raw_rot_rmse:.4f}°\n'
        f'EKF Odom → Trans RMSE: {ekf_trans_rmse:.4f} m,  '
        f'Rot RMSE: {ekf_rot_rmse:.4f}°'
    )
    ax.text(0.98, 0.02, rmse_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    # 儲存圖片
    out_dir = os.path.dirname(csv_path) or '.'
    out_png = os.path.join(out_dir, 'trajectory_comparison.png')
    fig.savefig(out_png, dpi=150)
    print(f'[INFO] 圖片已儲存 → {out_png}')

    plt.show()


if __name__ == '__main__':
    main()
