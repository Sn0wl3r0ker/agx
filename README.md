# AGX Hybrid Navigation System (ROS 1 Noetic + ROS 2 Humble)

![System Architecture](https://img.shields.io/badge/Architecture-Hybrid%20ROS1%2B2-blue) ![Platform](https://img.shields.io/badge/Platform-NVIDIA%20Jetson%20AGX%20Orin-green) ![Docker](https://img.shields.io/badge/Docker-Buildx%20Remote-blueviolet)

This is a **Docker-based** hybrid navigation system designed specifically for the **NVIDIA Jetson AGX Orin (JetPack 6)** platform. The project adopts modern DevOps workflows, enabling cross-compilation on a PC and one-click remote deployment to the edge device.


## 🏗️ System Architecture

The project utilizes a **dual-track architecture** with containerized isolation:

  * **`control` (ROS 1 Noetic)**: Low-level hardware drivers (Velodyne LiDAR, RealSense) and 3D SLAM.
  * **`planning` (ROS 2 Humble)**: High-level path planning (Nav2, Costmap) and AI extensions.
  * **`bridge`**: `ros1_bridge` for communication between legacy and modern ROS.
  * **`foxglove`**: WebSocket-based visualization (replacing RViz).
  * **`isaac_ros`**: GPU-accelerated perception nodes.

### 🌐 Network Topology

```mermaid
graph TD
    PC[PC Workstation] -- WiFi/SSH (192.168.200.x) --> AGX[AGX Orin]
    AGX -- Ethernet (192.168.1.x) --> LiDAR[Velodyne VLP-16]
    AGX -- USB --> Sensors[RealSense/Arduino]
```

-----

## 🚀 Quick Start (The "Makefile" Way)

We use a **Makefile** to automate complex Docker commands.

| Command | Description |
| :--- | :--- |
| **`make up`** | 🚀 **Start System**. Auto-detects PC/AGX mode. Fast boot. |
| **`make build`** | 🛠️ **Build Images**. Run this if you changed `Dockerfile`. |
| **`make rebuild`** | 🔄 **Rebuild + Start**. Clean restart after updates. |
| **`make down`** | 🛑 **Stop System**. Stops containers and removes networks. |
| **`make join`** | 🐳 **Enter Container**. Defaults to `isaac_ros`. |
| **`make logs`** | 📄 **View Logs**. Real-time logs from all services. |

-----

## 🛠️ Development Workflow

Choose the mode that fits your current task.

### Mode A: On-Device Development (Hot Reload)

***Best for:** Debugging, parameter tuning, and coding directly on the AGX.*

1.  Connect to AGX using **VS Code Remote - SSH**.
2.  Start the system:
    ```bash
    make up
    ```
3.  **Edit & Test**:
      * **Python**: Save file -\> `docker restart <container>` (or auto-reload if configured).
      * **C++**: Enter container -\> Compile.
        ```bash
        make join service=control
        ```

### Mode B: Remote Deployment (PC -\> AGX)

***Best for:** Clean builds, environment updates, and deploying from your powerful PC.*

1.  Switch your Docker context to the AGX:
    ```bash
    docker context use agx_remote
    ```
2.  Deploy (Build on PC, Run on AGX):
    ```bash
    # The Makefile detects the remote context and switches to AGX mode automatically
    make rebuild
    ```
3.  Switch back to local when done:
    ```bash
    docker context use default
    ```

-----

## 📊 Visualization (Foxglove Studio)

1.  **Open Foxglove Studio** (On PC).
2.  **Connection Setup**:
      * Source: `Foxglove WebSocket`
      * URL: `ws://<AGX_IP>:8765`
3.  **Troubleshooting**: If topics appear but no data shows, set QoS to **Reliable**.

-----

## 📝 Hardware Configuration

  * **LiDAR IP**: `192.168.1.201`
  * **AGX eth0 IP**: `192.168.1.77` (Must be static)
  * **Docker Network**: Uses `host` mode for maximum performance (no NAT).

-----

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

**Maintainer**: NYCUSystemLab