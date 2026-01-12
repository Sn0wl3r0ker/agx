# 參考網址
https://nvidia-isaac-ros.github.io/v/release-3.2/getting_started/hardware_setup/sensors/realsense_setup.html
https://nvidia-isaac-ros.github.io/v/release-3.2/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_segformer/index.html


# 權限問題
```bash
./scripts/run_dev.sh -d ${ISAAC_ROS_WS}
sudo chown -R admin:admin /workspaces/isaac_ros-dev
```
# 編譯依賴
```bash
rm -rf install/isaac_ros_unet build/isaac_ros_unet
colcon build --symlink-install \
--parallel-workers 2 \
--cmake-args -DBUILD_TESTING=OFF \
--packages-up-to isaac_ros_unet \
--allow-overriding gxf_isaac_atlas gxf_isaac_gxf_helpers gxf_isaac_messages gxf_isaac_sight isaac_ros_common isaac_ros_gxf isaac_ros_image_proc isaac_ros_managed_nitros isaac_ros_nitros isaac_ros_nitros_camera_info_type isaac_ros_nitros_image_type isaac_ros_tensor_list_interfaces isaac_ros_test
```
# Terminal 1
```bash
ros2 run realsense2_camera realsense2_camera_node --ros-args \
-p rgb_camera.profile:=640x480x30 \
-p align_depth.enable:=False \
-p enable_depth:=False \
-p enable_infra1:=False \
-p enable_infra2:=False
```

# 1. 確保目錄存在
``` bash
mkdir -p /workspaces/isaac_ros-dev/isaac_ros_assets/models/peoplesemsegnet/deployable_quantized_vanilla_unet_onnx_v2.0/1/
```
# 2. 開始編譯 (將 ONNX 轉為 TensorRT Engine)
```bash 
/usr/src/tensorrt/bin/trtexec \
  --onnx=/workspaces/isaac_ros-dev/isaac_ros_assets/models/peoplesemsegnet/deployable_quantized_vanilla_unet_onnx_v2.0/model.onnx \
  --saveEngine=/workspaces/isaac_ros-dev/isaac_ros_assets/models/peoplesemsegnet/deployable_quantized_vanilla_unet_onnx_v2.0/1/model.plan \
  --fp16 \
  --skipInference
ls -lh /workspaces/isaac_ros-dev/isaac_ros_assets/models/peoplesemsegnet/ deployable_quantized_vanilla_unet_onnx_v2.0/1/model.plan
```
# Terminal 2啟動
```bash 
ros2 launch isaac_ros_unet demo_final.launch.py
```