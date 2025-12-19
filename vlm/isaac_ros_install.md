# Isaac ROS 3.2 安裝與啟動指南

**適用環境：** NVIDIA Jetson AGX Orin
**系統版本：** JetPack 6.0 / 6.1 (Ubuntu 22.04)
**目標版本：** Isaac ROS 3.2 (release-3.2)

## 1\. 環境準備 (Host 端)

### 1.1 設定工作區目錄

建立專門存放 Isaac ROS 專案的資料夾（此處以你目前的自定義路徑 `~/disk/` 為例）：

```bash
mkdir -p ~/disk/workspaces/isaac_ros-dev/src
cd ~/disk/workspaces/isaac_ros-dev/src
```

### 1.2 安裝必要工具

確保 `git` 與 `git-lfs` 已安裝並初始化：

```bash
sudo apt-get install git-lfs
git lfs install --skip-smudge
```

-----

## 2\. 下載 Isaac ROS Common (關鍵步驟)

**⚠️ 重要：** JetPack 6 必須使用 `release-3.2` 分支，切勿使用 `main` (4.0/JP7) 或舊版。

```bash
cd ~/disk/workspaces/isaac_ros-dev/src
git clone -b release-3.2 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git
```

*(若要下載其他 GEMs 如 Visual SLAM，也必須使用 `-b release-3.2`)*

-----

## 3\. 修復 JetPack 6 Docker GPU 權限 (CDI 設定)

JetPack 6 改用 CDI (Container Device Interface) 管理 GPU，若缺少設定檔，Docker 會報錯 `unresolvable CDI devices`。

執行以下指令生成設定檔並重啟 Docker：

```bash
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
sudo systemctl restart docker
```

-----

## 4\. 啟動開發環境 (Docker)

### 4.1 執行啟動腳本

進入腳本目錄並執行 `run_dev.sh`。
**注意：** 因為工作區路徑不在預設位置，必須加上 `-d` 參數指定路徑。

```bash
cd ~/disk/workspaces/isaac_ros-dev/src/isaac_ros_common/scripts

# 第一次執行會花時間下載映像檔，請耐心等待
./run_dev.sh -d ~/disk/workspaces/isaac_ros-dev
```

### 4.2 驗證是否成功

當終端機提示符號變更為 `admin@tegra-ubuntu` (或類似名稱) 且位於 `/workspaces/isaac_ros-dev` 下，即代表成功進入容器。

檢查 GPU 是否掛載成功：

```bash
nvidia-smi
```

-----

## 5\. 容器內的操作 (常用的下一步)

進入容器後，通常需要安裝依賴並編譯程式碼：

```bash
# 1. 安裝 ROS 依賴
rosdep install -i -r --from-paths src --rosdistro humble -y

# 2. 編譯工作區
colcon build --symlink-install

# 3. 設定環境變數
source install/setup.bash
```

-----

### 常見問題筆記

1.  **錯誤：Specified isaac does not exist**
      * **解法：** 執行腳本時忘記加 `-d <你的路徑>`。
2.  **錯誤：failed to inject CDI devices**
      * **解法：** 忘記執行步驟 3 的 `nvidia-ctk cdi generate`。
3.  **分支混亂**
      * **解法：** 確保 `src` 下的所有 Isaac ROS 專案都是 `release-3.2` 分支。

-----
