import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import DiagGaussianDistribution

""" 階段訓練 """
STAGE1 = 64
STAGE2 = 128
STAGE3 = 128

# STAGE = STAGE1
STAGE = 256
# STAGE = STAGE3


""" 狀態空間模式 """
ORIGINAL_MODE = 11
ENERGY_MODE = 16

MODE = ORIGINAL_MODE
# MODE = ENERGY_MODE

class PreNet(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = STAGE, freeze_stage1: bool = False):
        super().__init__(observation_space, features_dim)
        self.observation_keys = list(observation_space.spaces.keys())
        self.freeze_stage1 = freeze_stage1
        self.subnets = nn.ModuleDict()
        input_dim = 0

        if "lidar" in self.observation_keys:
            self.subnets["lidar_net"] = nn.Sequential(

                # C=3, H=180, W=8
                nn.Conv2d(
                    in_channels=3, 
                    out_channels=8, 
                    kernel_size=(5,3), 
                    stride=(1,1), 
                    padding=(2,1)
                ),
                nn.BatchNorm2d(8),
                nn.LeakyReLU(0.01, inplace=True),

                # → (B, 8, 180, 8)                
                nn.Conv2d(
                    in_channels=8, 
                    out_channels=16, 
                    kernel_size=(5,3), 
                    stride=(1,1), 
                    padding=(2,1)
                ),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(0.01, inplace=True),

                # → (B, 16, 180, 10)
                nn.Conv2d(
                    in_channels=16, 
                    out_channels=24, 
                    kernel_size=(5,3), 
                    stride=(1,1), 
                    padding=(2,1)
                ),
                nn.BatchNorm2d(24),
                nn.ELU(inplace=True),

            )
            # feat_dim = 24 * 180 * 16  # 24*180*16
            # feat_dim = 24 * 180 * 10  # 24*180*10
            # feat_dim = 24 * 180 * 8  # 24*180*8
            feat_dim = 24 * 180 * 10  # 24*180*10


            self.subnets["lidar_lin"] = nn.Sequential(
                nn.Linear(feat_dim, 256),
                nn.LayerNorm(256),
                nn.ELU(),
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.ELU(),
            )
            input_dim += 128
        
        """
        state model
        """
        # state model for all states and actions
        state_dim = observation_space.spaces["state_current"].shape[0] if "state_current" in self.observation_keys else MODE # 默認狀態維度
        
        self.state_model = nn.Sequential(
            nn.Linear(state_dim, 128), nn.LayerNorm(128), 
            nn.LeakyReLU(negative_slope=0.01, inplace=True),

            nn.Linear(128, 64), nn.LayerNorm(64), 
            nn.LeakyReLU(negative_slope=0.01, inplace=True),

            nn.Linear(64, 64),  nn.LayerNorm(64), 
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
        input_dim += 64  # state_model 輸出 64 維

        self.fuse = nn.Sequential(
            nn.Linear(128 + 64, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
        )

    def forward(self, obs: dict) -> th.Tensor:
        features = []

        if "lidar" in self.observation_keys:
            lidar = obs["lidar"]  # 可能是 (B, N, 1) 或 (B, N)  # (B,180,8,3)

            x = lidar.permute(0,3,1,2) # (B, C=3, H=180, W=8)

            x = self.subnets["lidar_net"](x)         # → (B,24,180,8)
            x = x.flatten(1)   

            x = self.subnets["lidar_lin"](x)  # (B, 16) # → (B,64)

            features.append(x)

        if "state_current" in self.observation_keys:
            current_environment = self.state_model(obs["state_current"])  # (B, 64)
            features.append(current_environment)

        if not features:
            raise ValueError("No valid observation components provided")

        # 融合所有特徵
        out = self.fuse(th.cat(features, dim=1))  # (B, 128 or 192) -> (B, 128)
        # out = th.cat(features, dim=1)
        return out

class PPOTorchModel(ActorCriticPolicy):
    def __init__(self, observation_space: spaces.Dict, action_space: spaces.Box, lr_schedule, freeze_stage1: bool = False, **kwargs):
        self.input_dim = STAGE 
        kwargs["features_extractor_class"] = PreNet
        kwargs["features_extractor_kwargs"] = {"features_dim": self.input_dim, "freeze_stage1": freeze_stage1}
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

        action_dim = action_space.shape[0]
        self.actor = nn.Sequential(
            nn.Linear(self.input_dim, 128), nn.LayerNorm(128),
            nn.ELU(),
            nn.Linear(128, 64),nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, action_dim * 2)
        )

        self.critic = nn.Sequential(
            nn.Linear(self.input_dim, 128), nn.LayerNorm(128),
            nn.ELU(),
            nn.Linear(128, 64), nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # === 凍住 state_model 參數 ===
        if freeze_stage1:
            if hasattr(self.features_extractor, "state_model"):
                for param in self.features_extractor.state_model.parameters():
                    param.requires_grad = False
                print("[Freeze] State model parameters have been frozen.")
        
        # if freeze_stage1:
        #     # === 凍住 state_model 參數，只開最後一層 ===
        #     for name, param in self.features_extractor.state_model.named_parameters():
        #         # 只解凍最後一層 Linear + LayerNorm
        #         if "6" in name or "7" in name:
        #             param.requires_grad = True
        #         else:
        #             param.requires_grad = False

        #     print("=== Trainable parameters of state_model ===")
        #     for name, param in self.features_extractor.state_model.named_parameters():
        #         print(f"{name}: {param.requires_grad}")


        # Freeze 前兩個 blocks
        # if freeze_stage1:
        #     for layer in self.features_extractor.state_model[:6]:
        #         for param in layer.parameters():
        #             param.requires_grad = False

        #     # Unfreeze 最後一層 block (6,7,8)
        #     for layer in self.features_extractor.state_model[6:]:
        #         for param in layer.parameters():
        #             param.requires_grad = True

        # for param in self.actor.parameters():
        #     param.requires_grad = False
        # for param in self.critic.parameters():
        #     param.requires_grad = False

    def forward(self, obs: dict, state=None, deterministic: bool = False):
        features = self.extract_features(obs)
        dist_inputs = self.actor(features)
        mean_actions, log_std = th.chunk(dist_inputs, 2, dim=-1)
        
        self.action_dist.proba_distribution(mean_actions=mean_actions, log_std=log_std)
        value = self.critic(features).squeeze(-1)

        if deterministic:
            action = self.action_dist.get_actions(deterministic=True)
        else:
            action = self.action_dist.sample()

        log_prob = self.action_dist.log_prob(action)
        return action, value, log_prob
        # return action, log_prob, value

    def evaluate_actions(self, obs: dict, actions: th.Tensor):
        features = self.extract_features(obs)
        dist_inputs = self.actor(features)
        mean_actions, log_std = th.chunk(dist_inputs, 2, dim=-1)
        self.action_dist.proba_distribution(mean_actions=mean_actions, log_std=log_std)
        log_prob = self.action_dist.log_prob(actions)
        entropy = self.action_dist.entropy()

        value = self.critic(features).squeeze(-1)
        return value, log_prob, entropy

    def _predict(self, observation: dict, deterministic: bool = False) -> th.Tensor:
        features = self.extract_features(observation)
        dist_inputs = self.actor(features)
        mean_actions, log_std = th.chunk(dist_inputs, 2, dim=-1)
        self.action_dist.proba_distribution(mean_actions=mean_actions, log_std=log_std)
        if deterministic:
            return self.action_dist.get_actions(deterministic=True)
        return self.action_dist.sample()