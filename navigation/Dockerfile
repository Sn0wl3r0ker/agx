# syntax=docker/dockerfile:1.6

FROM nycusystemlab/l4t-noetic:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=noetic
ENV WORKSPACE=/root

# -------------------------------------------------
# 安裝 ROS 依賴與 GTSAM
# -------------------------------------------------
RUN apt-get update && apt-get install -y \
    ros-${ROS_DISTRO}-pcl-ros \
    libpcl-dev \
    ros-${ROS_DISTRO}-tf2-eigen \
    ros-${ROS_DISTRO}-message-filters \
    python3-catkin-tools \
    libpcap-dev \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

RUN add-apt-repository -y ppa:borglab/gtsam-release-4.1 && \
    apt-get update && \
    apt-get install -y libgtsam-dev libgtsam-unstable-dev && \
    rm -rf /var/lib/apt/lists/*

# -------------------------------------------------
# 建立工作目錄（空）# syntax=docker/dockerfile:1.6

# 繼承自您提供的 Base Image (已包含 ROS Noetic 基礎與 Cuda)
FROM nycusystemlab/l4t-noetic:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV WORKSPACE=/root

# -------------------------------------------------
# 1. 安裝 Realsense 與 GTSAM 需要的額外依賴
# -------------------------------------------------
# Base Image 已有 git/cmake/build-essential，這裡只補缺少的
RUN apt-get update && apt-get install -y \
    ros-${ROS_DISTRO}-pcl-ros \
    libpcl-dev \
    ros-${ROS_DISTRO}-tf2-eigen \
    ros-${ROS_DISTRO}-message-filters \
    ros-${ROS_DISTRO}-diagnostic-updater \
    ros-${ROS_DISTRO}-image-transport \
    libssl-dev \
    libusb-1.0-0-dev \
    libudev-dev \
    pkg-config \
    libpcap-dev \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------
# 2. 安裝 GTSAM
# -------------------------------------------------
RUN add-apt-repository -y ppa:borglab/gtsam-release-4.1 && \
    apt-get update && \
    apt-get install -y libgtsam-dev libgtsam-unstable-dev && \
    rm -rf /var/lib/apt/lists/*

# -------------------------------------------------
# 3. 手動編譯 Librealsense v2.48.0 (關鍵修復)
# -------------------------------------------------
# 鎖定 v2.48.0 以配合舊版 ROS code
# FORCE_RSUSB_BACKEND=ON: 讓 Docker 不需重新編譯 Kernel 即可抓到 USB
# BUILD_NETWORK_DEVICE=OFF: 解決缺少 fastcdr/fastrtps 的報錯
RUN git clone -b v2.48.0 https://github.com/IntelRealSense/librealsense.git /tmp/librealsense && \
    cd /tmp/librealsense && \
    mkdir build && cd build && \
    cmake ../ -DCMAKE_BUILD_TYPE=Release \
              -DBUILD_NETWORK_DEVICE=OFF \
              -DBUILD_EXAMPLES=OFF \
              -DBUILD_GRAPHICAL_EXAMPLES=OFF \
              -DFORCE_RSUSB_BACKEND=ON && \
    make -j$(nproc) && \
    make install && \
    rm -rf /tmp/librealsense

# -------------------------------------------------
# 4. 預先編譯缺失的 ROS 依賴
# -------------------------------------------------
# 解決 "Could not find package 'ddynamic_reconfigure'"
RUN mkdir -p /tmp/ros_deps_ws/src && \
    cd /tmp/ros_deps_ws/src && \
    git clone https://github.com/pal-robotics/ddynamic_reconfigure.git && \
    git clone https://github.com/ros-drivers/rgbd_launch.git && \
    cd /tmp/ros_deps_ws && \
    . /opt/ros/noetic/setup.sh && \
    catkin_make install -DCMAKE_INSTALL_PREFIX=/opt/ros/noetic && \
    rm -rf /tmp/ros_deps_ws

# -------------------------------------------------
# 5. 建立工作目錄 (包含 realsense_ws)
# -------------------------------------------------
RUN mkdir -p ${WORKSPACE}/hdl_ws/src \
             ${WORKSPACE}/lidar_ws/src \
             ${WORKSPACE}/realsense_ws/src

# -------------------------------------------------
# 6. 設定 Entrypoint
# -------------------------------------------------
COPY entrypoint.sh /root/entrypoint.sh
RUN chmod +x /root/entrypoint.sh

WORKDIR /root
ENTRYPOINT ["/root/entrypoint.sh"]
CMD ["bash"]
# -------------------------------------------------
RUN mkdir -p ${WORKSPACE}/hdl_ws/src \
             ${WORKSPACE}/lidar_ws/src

# -------------------------------------------------
# 複製 entrypoint
# -------------------------------------------------
COPY entrypoint.sh /root/entrypoint.sh
RUN chmod +x /root/entrypoint.sh

ENTRYPOINT ["/root/entrypoint.sh"]
CMD ["bash"]
