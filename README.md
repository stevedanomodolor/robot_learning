# Ball shooter robot learning

# Table of Contents
* [General Info](#General-info)
* [Code specification](#Code-specification)
* [Dependencies](#Dependencies)
* [Build](#Build)
* [Usage](#Usage)

## General Info
THe code contains an implementation of robot learning algorithm to teach a ball shooter to shoot a ball using the simulation gazebo

## Code specification
**TODO: Explain later**

## Dependencies
To execute the code, a python environment must be set to be able to interact with ros, to do so, use the following instructions
### Python 3
To run openai with ros you need to follow this steps here
https://www.youtube.com/watch?v=oxK4ykVh1EE
Steps

#### **Step 1**
1. Install basic python3 packages
```
sudo apt-get install python3-catkin-tools python3-dev python3-numpy
```
2. Install python virtualenv
```
sudo pip install virtualenv
```
3. Create and activate the virtual environment
```
mkdir -p python3_ws/src
cd python3_ws
```
4. Create the environment
```
virtualenv py3venv --python=python3
```
5. Activate the environment
```
source ~/python3_ws/py3venv/bin/activate
```

#### **Step 2**
```
cd ~/python3_ws/src
git clone https://github.com/openai/baselines.git
```
1. Baseline requires to install tensorflow
```
- pip install tensorflow
- pip install -e .
```
2 Then we install gym
```
pip install gym==0.15.4
```
You need this version for it to work with the baselines

#### **Step 3**
1. Install the ros package required by yout Ros code plus its dependecies in the pythin3 workspace
you need to install the code your python code needs and reinstall them in tpython 3
```
cd ~/python3_ws/src
git clone https://github.com/ros/geometry.git
git clone https://github.com/ros/geometry2.git
```
the dependencies of the previos package are the following
```
pip install pyaml
pip install rospkg
pip install empy
```

2. compile the catkin workspace:
```
cd ~/python3_ws
```
3.compile by using catkin build
**An error may occur**
-CMake Error at geometry/tf/CMakeLists.txt:87 (target_link_libraries):
  Attempt to add link library "tf" to target "test_transform_datatypes" which
  is not built in this directory.

  -This is allowed only when policy CMP0079 is set to NEW.
-fix

https://githubmemory.com/repo/ros/geometry/issues/213
```
- catkin_add_gtest(test_transform2_datatypes test/test_transform_datatypes.cpp)
- target_link_libraries(test_transform2_datatypes tf2  ${console_bridge_LIBRARIES})
- add_dependencies(test_transform2_datatypes ${catkin_EXPORTED_TARGETS})
```

#### **Step 4**
1. compile Your ros packages which depends on python3 libraries

open a new terminal with you catkin_ws- do this for evey terminal you open that require the python3
```
source ~/python3_ws/py3venv/bin/activate
```
**important**
2. make sure to delete your devel and build folder to start clean with python 3
```
rm -rf /build /devel
```
3. then build as follows
```
catkin_make -DPYTHON_EXECUTABLE:FILEPATH=/home/stevedan/python3_ws/py3venv/bin/python
```
4. then source as always in ros
```
source /devel/setup.bash
```

**important**
in any terminal you want to use this you need to activate the python environment
```
source ~/python3_ws/py3venv/bin/activate
```
## Build
1. To build, create a new catkin_ws workspace(xxxxx_ws->name the worksapce as you wish)
```
mkdir -p ~/xxxxx_ws/src
```
2. git clone this repository to the src folder
```
cd ~/xxxxx_ws/src
git clone https://github.com/stevedanomodolor/robot_learning.git
```
3. go back to the xxxxx_ws folder and build
```
cd ~/xxxxx_ws
catkin_make
```
## Usage
To run the code,
1. Open three terminal and in each of the terminals activate the python environment and source your worksapce where the ball shooter code is
 ```
 cd ~/xxxxx_ws
 source ~/python3_ws/py3venv/bin/activate
 source ~/xxxxx_ws/devel/setup.bash
 ```
 2. Make sure to chmod the python files in the ball_shooter_training/scripts folder(it is not necessary to do all but cant remember which so :)
 ```
 chmod +x xxxxxxxxxxxx.py
 ```
 3. In the first terminal launch the simulation, you might not see the ball, it is normal
 ```
roslaunch ball_shooter_sim main.launch
 ```
 4. In the second terminal, launch the vision node
 ```
rosrun ball_shooter_training vision_main.py
 ```
 5. In the last terminal launch the training node
 ```
roslaunch ball_shooter_training main.launch  
 ```
