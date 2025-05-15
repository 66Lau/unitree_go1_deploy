# Unitree Go1 Deployment in Jetson

## Installation

### 1. SDK(with LCM forward)

```bash
sudo apt update
sudo apt install libboost-all-dev
# install LCM
sudo apt install liblcm-dev

# complie sdk
cd unitree_legged_sdk
mkdir build
cmake ..
make
```

### 2. Deployment code

If Jetson not install conda yet:

```bash
# open a new terminal and back to home
cd ~

# 1. Download ARM64  Miniforge3 sh file
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh

# 2. assign permission
chmod +x Miniforge3-Linux-aarch64.sh

# 3. install
# Follow the installation tutorial in the terminal
./Miniforge3-Linux-aarch64.sh

# 5. Test installation
source ~/.bashrc
conda --version

```

Install pytorch first, we don't recommand using pip install torch directly, might get x86 version if you are using jetson.

Go and check https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048 and find matched version pytorch wheel for your system.

If you don't know your system jetpack, use the command `cat /etc/nv_tegra_release`. My jetpack is 5.1, 

```bash
# create environment and activate
conda create -n go1_deploy python=3.8
conda activate go1_deploy

sudo apt install libopenblas-base libopenblas-dev

# find pytorch wheel of your system and replace
wget https://developer.download.nvidia.com/compute/redist/jp/v502/pytorch/torch-1.13.0a0+d0d6b1f2.nv22.10-cp38-cp38-linux_aarch64.whl
# install
pip install torch-1.13.0a0+d0d6b1f2.nv22.10-cp38-cp38-linux_aarch64.whl

```

Install the environment
```bash
# install
cd go1_deploy
pip install -e .
```
---

## Run Policy
- `Ping 192.168.123.15` to validate connection
- Press `L2 + A`, make robot get down
- Press `L2 + B`, enter damping mode

1. Deploy in AGX
```bash
# Start communication with Go1
cd unitree_legged_sdk/build
./lcm_position
# Open a new terminal, launch policy
conda activate go1_deploy
cd go1_deploy/go1_gym_deploy/scripts/
# deploy a well-trained walking policy
python deploy.py
```

2. Or deploy in go1 embedded jetson
```bash
# first go right branch
rsync -av --exclude='build' ../unitree_go1_deploy unitree@192.168.123.15:~
ssh unitree@192.168.123.15
cd ~/unitree_go1_deploy/unitree_legged_sdk/build
./lcm_position
# new terminal
cd ~/unitree_go1_deploy/go1_deploy/go1_gym_deploy/scripts/
python3 deploy.py
```


# External Perception
This part only related to our own hardware setting (Xvaier AGX + D435 + T265), environment is Ros2 Foxy on Ubuntu 20.04

Make sure the fan of AGX is running in high speed
```
sudo /usr/bin/jetson_clocks --fan
```

Runing Rosbridge for visualization in local network
```bash
# sudo apt install ros-foxy-image-transport-plugins
# This important for a automatically compress image, otherwise full resolution image would lead to high delay
conda deactivate
ros2 launch rosbridge_server rosbridge_websocket_launch.xml
```

Start D435 + T265 topic node
```bash
# Recognize serial number of different camera
rs-enumerate-devices -s
# launch D435
ros2 launch realsense2_camera rs_launch.py serial_no:="'827312072741'"
# launch T265
ros2 launch realsense2_camera rs_launch.py serial_no:="'146322110342'"
```



# Websocket Relay

### In Server
For the communication with server to achieve high level policy inference

- Subscrible go1 perception info
```bash
# In Server
sudo apt install ros-humble-compressed-image-transport
cd websocket
conda deactivate
python3 relay2server.py
```

- Visualization in server rviz
```bash
# In Server Visualization
rviz2 -d websocket/visualize.rviz
```

- Send expected velocity to relay
```bash
# if wifi is open 
sudo ip route add    224.0.0.0/4 dev eno1    metric 100
python3 server2relay.py
ros2 launch rosbridge_server rosbridge_websocket_launch.xml
```

### In Relay

- Relay Camera information
```bash
ros2 launch rosbridge_server rosbridge_websocket_launch.xml
ros2 launch realsense2_camera rs_launch.py serial_no:="'827312072741'"
ros2 launch realsense2_camera rs_launch.py serial_no:="'146322110342'"
# or 
ros2 launch realsense2_camera rs_launch.py depth_module.profile:=640x480x30 rgb_camera.profile:=640x480x30 align_depth.enable:=true serial_no:="'827312072741'"
ros2 launch realsense2_camera rs_launch.py serial_no:="'146322110342'"
```

- Receive velocity message from server and relay to Go1
```bash
conda deactivate
cd websocket
python3 relay2go1.py
```

### Final Command

Perception
```bash
# 1
ros2 launch rosbridge_server rosbridge_websocket_launch.xml
# 2
python3 websocket/relay2server_socket.py
# 3
ros2 launch realsense2_camera rs_launch.py depth_module.profile:=640x480x30 rgb_camera.profile:=640x480x30 align_depth.enable:=true serial_no:="'827312072741'"
#4
ros2 launch realsense2_camera rs_launch.py serial_no:="'146322110342'"

```


Control
```bash
# 1.
sudo ip route add    224.0.0.0/4 dev eno1    metric 100
python3 websocket/relay2go1.py

# 2. 
ssh unitree@192.168.123.15
cd ~/unitree_go1_deploy/unitree_legged_sdk/build
./lcm_position

# 3.
ssh unitree@192.168.123.15
cd ~/unitree_go1_deploy/go1_deploy/go1_gym_deploy/scripts/
python3 deploy.py



```