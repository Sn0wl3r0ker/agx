# ./vlm/Dockerfile

# 1. 基底映像檔
FROM nvcr.io/nvidia/isaac/ros:aarch64-ros2_humble_4c0c55dddd2bbcc3e8d5f9753bee634c

ENV DEBIAN_FRONTEND=noninteractive

# 2. 安裝基礎工具與 RealSense 驅動
RUN apt-get update && apt-get install -y \
    python3-pip \
    usbutils \
    ros-humble-realsense2-camera \
    ros-humble-diagnostic-updater \
    ros-humble-xacro \
    git-lfs \
    vim \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install -U jetson-stats

# 4. 安裝 Isaac ROS 相關套件
RUN curl -fsSL https://dl.yarnpkg.com/debian/pubkey.gpg | gpg --dearmor -o /usr/share/keyrings/yarn.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/yarn.gpg] https://dl.yarnpkg.com/debian stable main" > /etc/apt/sources.list.d/yarn.list

RUN apt-get update && apt-get install -y \
    ros-humble-isaac-ros-common \
    ros-humble-isaac-ros-apriltag \
    ros-humble-isaac-ros-image-pipeline \
    ros-humble-isaac-ros-nitros \
    ros-humble-isaac-ros-jetson-stats \
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-isaac-ros-peoplesemseg-models-install \
    ros-humble-vision-msgs \
    && apt-get remove -y ros-humble-compressed-depth-image-transport \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# 5. Grounding DINO 依賴 (跨版本移植 4.x → 3.2)
#    - transformers: 文字 tokenizer 需要
RUN pip3 install transformers

# 6. cuda_stream.hpp 相容性 shim (4.x API → 3.2)
#    isaac_ros_common 3.2 缺少此 header，由 shim 提供 inline 實作
COPY shims/cuda_stream.hpp /tmp/cuda_stream_shim.hpp
RUN COMMON_INCLUDE=$(find /opt/ros/humble -path '*/isaac_ros_common' -type d | head -1) && \
    if [ -n "$COMMON_INCLUDE" ]; then \
      cp /tmp/cuda_stream_shim.hpp "$COMMON_INCLUDE/cuda_stream.hpp"; \
      echo "Installed cuda_stream.hpp shim to $COMMON_INCLUDE"; \
    fi && \
    rm -f /tmp/cuda_stream_shim.hpp

# 7. Git safe directory (避免 volume mount 的 ownership 錯誤)
RUN git config --global --add safe.directory '*'

# 8. 設定工作目錄
WORKDIR /workspaces/isaac_ros-dev
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# 9. 把 source setup.bash 寫入
RUN echo "source /workspaces/isaac_ros-dev/install/setup.bash" >> /root/.bashrc