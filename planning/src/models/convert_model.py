import sys
import numpy as np

# ==============================================================================
# 步驟 1: 先 Import 所有依賴庫
# 讓 Pandas 和 SB3 在 "乾淨" 的 NumPy 1.26 環境下初始化
# 這樣 Pandas 就不會被後面的補丁騙到而崩潰
# ==============================================================================
print("1. Importing libraries (Pandas/SB3)...")
from stable_baselines3 import PPO
import torch
print("✅ Libraries imported successfully.")

# ==============================================================================
# 步驟 2: 載入後才打補丁 (Late Patching)
# 這是為了騙過 PPO.load() 裡面的 pickle 反序列化器
# ==============================================================================
print(f"2. Applying NumPy 2.0 patch for model loading (Current: {np.__version__})...")

try:
    # 抓取舊版物件
    from numpy import core
    from numpy.core import multiarray
    from numpy.core import numeric
    
    # 建立假路徑 (這時候 Pandas 已經載入完了，不會再受影響)
    sys.modules['numpy._core'] = core
    sys.modules['numpy._core.multiarray'] = multiarray
    sys.modules['numpy._core.numeric'] = numeric
    
    print("✅ Patch applied. Ready to load legacy model.")

except ImportError as e:
    print(f"❌ Patch failed: {e}")
    sys.exit(1)

# ==============================================================================
# 步驟 3: 轉換模型
# ==============================================================================
input_model = "best_model.zip"
output_model = "best_model_1x.zip"

print(f"3. Converting {input_model} -> {output_model} ...")

try:
    # 載入 (Pickle 會用到上面的補丁)
    # 如果你有用自定義 Policy，記得加 custom_objects
    # model = PPO.load(input_model, custom_objects={"policy_class": PPOTorchModel})
    model = PPO.load(input_model)
    print("   Model loaded into memory!")
    
    # 存檔 (因為現在環境是 1.26，會自動存成舊版相容格式)
    model.save(output_model)
    print(f"🎉 Success! Saved to: {output_model}")
    print("   Please update your inference.py to use this new file.")

except Exception as e:
    print(f"❌ Error: {e}")
    # 印出更多細節幫助除錯
    import traceback
    traceback.print_exc()