# AGX Hybrid Navigation System (ROS 1 Noetic + ROS 2 Humble)

![System Architecture](https://img.shields.io/badge/Architecture-Hybrid%20ROS1%2B2-blue) ![Platform](https://img.shields.io/badge/Platform-NVIDIA%20Jetson%20AGX%20Orin-green) ![Docker](https://img.shields.io/badge/Docker-Buildx%20Remote-blueviolet)

This is a **Docker-based** hybrid navigation system designed specifically for the **NVIDIA Jetson AGX Orin (JetPack 6)** platform. The project adopts modern DevOps workflows, enabling cross-compilation on a PC and one-click remote deployment to the edge device.


## 🏗️ System Architecture

The project utilizes a **dual-track architecture** with containerized isolation:

| Container | Role & Description |
| --- | --- |
| **`control`** | **[ROS 1 Noetic]** Handles low-level hardware drivers (Velodyne LiDAR, RealSense) and 3D SLAM algorithms. |
| **`bridge`** | **[ROS 1 Bridge]** A dedicated bridge using `ros1_bridge` to enable seamless topic communication between Noetic and Humble. |
| **`planning`** | **[ROS 2 Humble]** Responsible for high-level path planning (Nav2, Costmap) and behavior trees. |
| **`foxglove`** | **[Visualization]** Runs a high-performance WebSocket server for remote visualization (replaces the heavy RViz client). |
| **`isaac_ros`** | **[Perception]** Leverages NVIDIA Isaac ROS for GPU-accelerated VSLAM and AI perception tasks. |


### 🌐 Network Topology

```mermaid
graph TD
    PC[PC Workstation] -- WiFi/SSH (192.168.200.x) --> AGX[AGX Orin]
    AGX -- Ethernet (192.168.1.x) --> LiDAR[Velodyne VLP-16]
    AGX -- USB --> Sensors[RealSense/Arduino]
```

-----

## 🚀 Quick Start (The "Makefile" Way)

### 1. Build and Start Services

Initialize the Docker environment. The system automatically detects your architecture (AGX vs PC).

```bash
# Build docker images
make build
# Start services in the background
make up
```

### 2. Enter the Environment

To open a terminal inside a running container (e.g., `control`, `planning`):

```bash
make join
```

*This command opens an interactive menu to select the target container.*

### 3. Launch ROS Tasks (Task Manager)

To execute predefined hardware tasks (e.g., Lidar, Camera, Control) in the background:

```bash
make run
```

*Select a task from the menu. The task will run in a detached Tmux session.*

---

## 🛠️ Makefile Workflow

This project uses a "Fire and Forget" workflow for running ROS nodes.

| Command | Description |
| --- | --- |
| **`make up`** | Start all containers (`control`, `planning`, `visualization`, etc.) in detached mode. |
| **`make run`** | **Launch a ROS Task.** Opens a menu to start nodes like *Keyboard Control*, *Lidar*, or *SLAM*. |
| **`make view`** | **Monitor a Task.** Attach to the Tmux session of a running background task to see logs or control the robot. |
| **`make stop`** | **Terminate a Task.** Select specific background tasks to kill. |
| **`make join`** | Open a bash shell in a specific container (Interactive selection). |
| **`make logs`** | Follow the logs of all Docker services. |
| **`make rebuild`** | Force rebuild images and restart containers (useful after Dockerfile changes). |
| **`make down`** | Stop and remove all containers. |


## 📊 Visualization (Foxglove Studio)



This project uses **Foxglove Studio** instead of RViz for remote monitoring.



1.  **Open Foxglove Studio** (On PC).

2.  **Connection Setup**:

      * Source: `Foxglove WebSocket`

      * URL: `ws://<AGX_IP>:8765` (AGX WiFi IP)

3.  **Common Topics**:

      * `Map`: `/globalmap` (PointCloud2)

      * `LiDAR`: `/velodyne_points` (PointCloud2)

      * `Path`: `/global_path` (MarkerArray)

      * `Robot`: `/tf`



> **Tip**: If connected but no data appears, check if the Topic QoS settings in Foxglove are set to **Reliable**



-----



## 📝 Hardware Notes



### Velodyne LiDAR Setup



The LiDAR uses Ethernet UDP. You must configure the AGX's wired interface (`eth0`) to a separate subnet.



  * **LiDAR IP**: `192.168.1.201` (Default)

  * **AGX eth0 IP**: `192.168.1.x` (Manual Static IP, e.g., 77)

  * **Docker Port Mapping**: `2368:2368/udp`



-----



## 🗓️ Roadmap



  - [x] **Phase 1**: Establish AGX JetPack 6 Hybrid Architecture (ROS 1 + ROS 2)

  - [x] **Phase 2**: Implement Buildx Remote Deployment Workflow

  - [x] **Phase 3**: Integrate Hardware Drivers (Velodyne, RealSense) & Docker Network Passthrough

  - [x] **Phase 4**: Replace RViz with Foxglove Studio for Web-based Viz

  - [ ] **Phase 5**: Deploy Nav2 Stack and bridge with SLAM maps

  - [ ] **Phase 6**: Integrate VLM/RL models into ROS 2 nodes for AI Navigation

## 👥 For New Team Members (One-Time Setup)

If you are new to the project, follow these steps to set up your PC for remote deployment.

### 1\. Generate SSH Key (On PC)

skip this if you already have one.

```bash
ssh-keygen -t ed25519 -C "your_name@pc"
```

### 2\. Authorize Key on AGX

Ask the lead for the AGX password, then copy your key:

```bash
ssh-copy-id systemlabagx@<AGX_IP>
```

### 3\. Create Docker Remote Context

This allows your local Docker CLI to control the Docker Engine on the AGX.

```bash
# Replace <AGX_IP> with the actual IP (e.g., 192.168.200.112)
docker context create agx_remote --docker "host=ssh://systemlabagx@<AGX_IP>"

# Verify connection
docker --context agx_remote info
```
### Remote Deployment (PC -\> AGX)

***Best for:** Clean builds, environment updates, and deploying from powerful PC.*

1.  Switch your Docker context to the AGX:
    ```bash
    docker context use agx_remote
    ```
2.  Switch back to local when done:
    ```bash
    docker context use default
    ```
> **Note**: In PC mode, containers use the code baked into the Docker Image. Local source files on the AGX are **not** mounted.
-----

**Maintainer**: NYCUSystemLab