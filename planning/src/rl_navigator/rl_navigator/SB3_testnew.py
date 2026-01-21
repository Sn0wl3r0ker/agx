# SB3.py  — Single-env training + evaluation (no EvalCallback)

import os
import time
import logging
import argparse
import torch
import gymnasium as gym

from stable_baselines3 import PPO   # PPO 演算法
from stable_baselines3.common.env_util import make_vec_env  # 產生 VecEnv(n_envs=1)
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback # 自訂 callback & 存檔 callback 基底
from stable_baselines3.common.evaluation import evaluate_policy # SB3 官方評估函式
from stable_baselines3.common.vec_env import VecNormalize   # VecNormalize 觀測/回報正規化 wrapper
from stable_baselines3.common.save_util import load_from_zip_file # 載入模型用的工具

from GPT5_gazebo import GazeboEnv   # 自訂 Gazebo RL env
from ppo_nn import PPOTorchModel    # 自訂 policy (Actor-Critic NN)

# =========================================================
#  Callback: Evaluate on the SAME train env
# =========================================================
class EvalOnTrainEnvCallback(BaseCallback): # Eval on train env
    """
    Evaluate on the SAME env to keep program-driven dynamic obstacles alive.
    """
    def __init__(
        self,
        train_env,              # 你傳進來的 env
        eval_every_steps: int,  # 你傳進來的參數
        n_eval_episodes: int,   # 每次評估跑幾個 episode
        best_model_save_path: str,  # 最佳模型存放路徑
        vecnorm_path: str = None,   # VecNormalize 存檔路徑（如果有的話）
        deterministic: bool = True, # 評估時是否用確定性動作
        verbose: int = 1,   # 是否印出評估結果
    ):
        super().__init__(verbose)       # 初始化 BaseCallback（會建立 self.model 等屬性）
        self.train_env = train_env      # 保存 env 引用（注意：這是同一個 env instance）
        self.eval_every_steps = max(1, int(eval_every_steps))   # 轉成 int，保險
        self.n_eval_episodes = int(n_eval_episodes)             # 轉成 int，保險
        self.best_model_save_path = best_model_save_path        # 保存 best model 的資料夾
        self.vecnorm_path = vecnorm_path                        # 保存 vecnorm pkl 的路徑
        self.deterministic = deterministic                      # 保存 deterministic 設定
        self.best_mean_reward = float("-inf")                   # 初始化最佳平均獎勵為負無限大

        os.makedirs(self.best_model_save_path, exist_ok=True)   # 確保 best_model 資料夾存在

    def _on_step(self) -> bool:       # SB3 在每個 environment step 會呼叫一次
        # ---- 若還沒到評估時機，直接回 True 繼續訓練 ----
        if self.num_timesteps % self.eval_every_steps != 0:   # num_timesteps 是 SB3 內建累積步數
            return True                                       # True 表示「不要中止訓練」
        
        # ====================================================
        # 1) 存 VecNormalize 統計（非常重要）
        #    - VecNormalize.save() 會把 running mean/var 等存到 pkl
        #    - 你 restore/resume 時才會「看到同樣的 normalize 空間」
        # ====================================================
        if self.vecnorm_path is not None:                   # 確認你有給路徑
            try:
                self.train_env.save(self.vecnorm_path)      # 對 VecNormalize wrapper 直接存 pkl
            except Exception as e:
                print(f"[Eval] VecNorm save failed: {e}")   # 存檔失敗也不要中止訓練

        # ====================================================
        # 2) evaluate_policy
        #    - 這會呼叫 env.reset() 跑 n_eval_episodes 次 episode
        #    - 因為你用同一個 env instance，動態障礙仍在同一個 Gazebo 裡
        #    - 注意：evaluate_policy 會用你此刻 env 的 training/norm_reward 狀態
        #      你若想「評估不更新 VecNorm 統計」，就要在這裡暫時切 env.training=False
        #      但你工程上想保持一致環境狀態，所以這裡採「不切」做法（可接受）
        # ====================================================
        mean_reward, std_reward = evaluate_policy(          # 回傳平均/標準差 reward
            self.model,                                     # SB3 會把 model 注入 callback
            self.train_env,                                 # 用同一個 env instance 評估
            n_eval_episodes=self.n_eval_episodes,           # 評估 episodes 數
            deterministic=self.deterministic,               # 是否 deterministic
            render=False,
        )
        # ====== TensorBoard logging ======
        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.record("eval/std_reward", std_reward)
        self.logger.record("eval/n_eval_episodes", self.n_eval_episodes)
        self.logger.dump(step=self.num_timesteps)




        # ---- 印出評估結果 ----
        if self.verbose:
            print(
                f"[EvalOnTrainEnv] steps={self.num_timesteps} "
                f"mean={mean_reward:.3f} std={std_reward:.3f}"
            )

        # ====================================================
        # 3) 若創新高 → 存 best model
        #    - 存的是 policy/optimizer 等（SB3 的 zip）
        # ====================================================
        if mean_reward > self.best_mean_reward:                  # 若本次 mean 超過歷史最佳
            self.best_mean_reward = mean_reward                   # 更新最佳值
            save_path = os.path.join(self.best_model_save_path, "best_model")    # best_model 檔名前綴
            self.model.save(save_path)                                  # 存成 best_model.zip
            print(f"[EvalOnTrainEnv] New best model saved: {save_path}")

        return True                  # True 代表訓練繼續


# =========================================================
#  Paths / config
# =========================================================
BASE_LOGDIR = os.path.abspath("./sb3_results_0116_stage1_generalization/ppo")   # 本次實驗資料夾（含 best、checkpoints、vecnorm）
TENSORBOARD_LOGDIR = os.path.join(BASE_LOGDIR, "tensorboard")                   # TensorBoard log 路徑
VECNORM_PATH = os.path.join(BASE_LOGDIR, "vecnormalize.pkl")                    # VecNormalize 統計存檔（最重要那個 pkl）                 
PRETRAINED_MODEL_PATH = "/home/systemlab/test_RL_ws/src/adaptiveON/src/my_git/sb3_results_0107_stage1_generalization/ppo/best_model/best_model.zip" # stage1 存檔點(new , choice)
# PRETRAINED_MODEL_PATH = "/home/systemlab/test_RL_ws/src/adaptiveON/src/my_git/sb3_results_0112_stage1_generalization/ppo/checkpoints/ppo_260000_steps.zip" # stage1 存檔點(new , choice)

STAGE1 = 1
STAGE2 = 2
STAGE = STAGE2


# =========================================================
#  Env builder (ONLY ONE ENV)
# =========================================================
def make_train_env():
    def _make():
        if STAGE == STAGE1:
            obs = ["state", "action"]
        else:
            obs = ["lidar", "state", "action"]

        return GazeboEnv(
            launchfile="/home/systemlab/Lun_ws/src/six_wheels_vehicle_3/launch/gazebo_rviz.launch",
            observation_components=obs,
        )

    env = make_vec_env(_make, n_envs=1)
    # env = VecNormalize(env, norm_obs=True, norm_reward=True)
    return env


# =========================================================
#  Main
# =========================================================
def main(args):
    os.makedirs(BASE_LOGDIR, exist_ok=True)
    os.makedirs(TENSORBOARD_LOGDIR, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        filename="sb3_gazebo.log",
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

# ========================================================
# Restore-only 模式（只載入模型並評估）
# ========================================================
    if args.restore:                    # 若使用者下了 --restore
        # 1. 建立 env（與訓練時完全相同）
        env = make_train_env()           # 1) 先建立 base env（VecEnv）

        # env = VecNormalize.load(VECNORM_PATH, env)
        # ----------------------------------------------------
        # 2. 載入 VecNormalize 統計（如果存在）
        # ----------------------------------------------------
        if os.path.exists(VECNORM_PATH):    # 檢查 pkl 是否存在
            env = VecNormalize.load(VECNORM_PATH, env)       # 用舊統計包起來（順序必須在 load model 之前）
            env.training = False        # restore / eval 不更新統計
            env.norm_reward = False     # eval 時看「真實 reward」而不是 normalized reward
            print(f"[RESTORE] VecNormalize loaded from {VECNORM_PATH}")     # 印提示
        else:
            print("[RESTORE] VecNormalize not found, using fresh stats")

        # ----------------------------------------------------
        # 3. 決定要載入的模型路徑
        #    - CLI 有給就用 CLI
        #    - CLI 沒給就用預設 best_model
        # ----------------------------------------------------
        if args.model_path != "":
            model_path = args.model_path
            print(f"[RESTORE] Using model from CLI: {model_path}")
        else:
            model_path = os.path.join(
                BASE_LOGDIR,
                "best_model",
                "best_model.zip"
            )
            print(f"[RESTORE] Using default best model: {model_path}")

        # ----------------------------------------------------
        # 4. 確認模型檔案真的存在
        # ----------------------------------------------------
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"[RESTORE] Model file not found: {model_path}"
            )

        # ----------------------------------------------------
        # 5. 載入模型
        # ----------------------------------------------------
        model = PPO.load(
            model_path,
            env=env,
            device=device
        )

        print("[RESTORE] Model loaded successfully")

        # ----------------------------------------------------
        # 6. 執行評估（會 reset env，但仍是同一個 env instance）
        # ----------------------------------------------------
        mean_reward, std_reward = evaluate_policy(  # 跑 evaluate
            model,                                  # 模型
            env,                                    # env（含 VecNormalize）
            n_eval_episodes=5,                      # 評估幾個 episode
            deterministic=True                      # deterministic 評估
        )

        print(
            f"[RESTORE RESULT] mean_reward={mean_reward:.3f}, "
            f"std_reward={std_reward:.3f}"
        )

        # ----------------------------------------------------
        # 7. 結束程式（restore 模式不進入訓練）
        # ----------------------------------------------------
        env.close()   # 關閉 env（Gazebo 可能也會收尾）
        return      # 結束程式（restore 模式不訓練）


    # -----------------------------
    # Training MODE
    # -----------------------------
    env = make_train_env()

    # ---- VecNormalize: new or resume ----
    if args.resume and os.path.exists(VECNORM_PATH):    # 若 --resume 且 vecnorm.pkl 存在
        env = VecNormalize.load(VECNORM_PATH, env)      # 載入舊統計（保持 obs/return scale 一致）
        env.training = True                     # 續訓時允許繼續更新統計
        env.norm_reward = True                  # 訓練時通常 reward normalize = True
        print("[TRAIN] VecNormalize restored")  # 印提示
    else:
        env = VecNormalize(                     # 新建 VecNormalize wrapper
            env,                                # 包住 VecEnv
            norm_obs=True,                      # observation 正規化
            norm_reward=True,                   # reward 正規化
            clip_obs=10.0,                      # obs clip，避免極端值破壞網路
        )
        print("[TRAIN] VecNormalize initialized")

    # 3) 建立 PPO 模型（注意：此時 env 已經是 VecNormalize wrapper）
    model = PPO(
        policy=PPOTorchModel,               # 你的自訂 Actor-Critic policy
        env=env,                            # 包含 VecNormalize 的 env
        learning_rate=1e-4,                 # learning rate
        n_steps=2048,                       # rollout 長度
        batch_size=128,                     # minibatch size
        n_epochs=10,                        # 每次 update 的 epochs
        gamma=0.99,                         # 折扣因子
        gae_lambda=0.95,                    # GAE lambda
        clip_range=0.2,                     # PPO clip range
        ent_coef=0.01,                      # entropy bonus coefficient
        vf_coef=0.5,                        # value function loss coefficient
        max_grad_norm=0.5,                  # gradient clipping
        normalize_advantage=True,           # advantage normalize
        target_kl=0.02,                     # 目標 KL 散度
        tensorboard_log=TENSORBOARD_LOGDIR, # TB log
        device=device,                      # 使用 CPU 或 GPU
        verbose=1,                          # 印訓練過程
    )

    # ---- resume partial load (exclude lidar*) ----
    if args.resume and os.path.exists(PRETRAINED_MODEL_PATH):   # 若 resume 且 pretrained 存在
        _, params, _ = load_from_zip_file(PRETRAINED_MODEL_PATH, device=device)  
        old_sd = params["policy"]
        new_sd = model.policy.state_dict()

        # 遍歷新模型的每個參數 key
        for k in new_sd:
            if k in old_sd and not k.startswith("lidar"): # 若舊模型也有這個 key 且不是 lidar 開頭（你指定排除 lidar 分支）
                if new_sd[k].shape == old_sd[k].shape:    # 確認 shape 一致
                    new_sd[k] = old_sd[k]                 # 用舊權重覆蓋新權重

        model.policy.load_state_dict(new_sd, strict=False)
        print("[Resume] partial weight loaded (exclude lidar*)")

    # ---- callbacks ----
    checkpoint_cb = CheckpointCallback(     # 定期存 checkpoints（方便中斷後續訓）
        save_freq=10000,                    # 每 10000 steps 存一次
        save_path=os.path.join(BASE_LOGDIR, "checkpoints"),     # checkpoints 存放資料夾
        name_prefix="ppo",
    )

    eval_cb = EvalOnTrainEnvCallback(       # 你的自訂 eval callback（同 env）
        train_env=env,                      # 傳入同一個 env instance
        eval_every_steps=20000,             # 每 20000 steps 評估一次
        n_eval_episodes=3,                  # 每次評估跑 3 個 episode
        best_model_save_path=os.path.join(BASE_LOGDIR, "best_model"),
        vecnorm_path=VECNORM_PATH,
        deterministic=True,                 # 評估時用確定性動作
        verbose=1,

    )

    # ---- learn ----
    model.learn(
        total_timesteps=1_000_000,
        callback=[checkpoint_cb, eval_cb],
        progress_bar=True,
    )

    env.save(VECNORM_PATH)
    env.close()


# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model_path", type=str, default=PRETRAINED_MODEL_PATH)
    args = parser.parse_args()

    main(args)
