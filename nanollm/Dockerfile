# ./nanollm/Dockerfile

# 1. 基底映像檔 - Jetson 專用 NanoLLM (含 ROS Humble)
FROM dustynv/nano_llm:humble-r36.3.0

ENV DEBIAN_FRONTEND=noninteractive

# 2. 安裝基礎工具
RUN apt-get update && apt-get install -y \
    python3-pip \
    curl \
    wget \
    git \
    htop \
    nano \
    && rm -rf /var/lib/apt/lists/*

# 3. 安裝 Jetson 監控工具
RUN pip3 install -U jetson-stats --index-url https://pypi.jetson-ai-lab.io/jp6/cu126

# 4. 安裝額外 ROS 2 Humble 套件（基底已包含 ROS Humble）
RUN apt-get update && apt-get install -y \
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    && rm -rf /var/lib/apt/lists/*

# 5. 安裝額外的 Python 套件
RUN pip3 install --no-cache-dir --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
    opencv-python \
    pillow \
    numpy

# 6. 設定環境變數
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ENV ROS_DOMAIN_ID=0

# 7. 設定工作目錄
WORKDIR /data

# 8. 把 source setup.bash 寫入
RUN echo "source /opt/ros/install/setup.bash" >> /root/.bashrc

# 9. 預設指令
CMD ["bash", "-l"]
