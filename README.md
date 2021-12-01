#important
To run openai with ros you need to follow this steps here
https://www.youtube.com/watch?v=oxK4ykVh1EE
Steps
Step 1
1. Install basic python3 packages
sudo apt-get install python3-catkin-tools python3-dev python3-numpy

2. Install python virtualenv
sudo pip install virtualenv

3. Create and activate the virtual environment
mkdir -p python3_ws/src
cd python3_ws
#Create the environment
virtualenv py3venv --python=python3
#activate the environment
source ~/python3_ws/py3venv/bin/activate

Step 2
cd ~/python3_ws/src
git clone https://github.com/openai/baselines.git
baseline require to install tensorflow
pip install tensorflow
pip install -e .
#then we install gym
pip install gym==0.15.4
You need this version for it to work with the baselines

Step 3 install the ros package required by yout Ros code plus its dependecies in the pythin3 workspace
you need to install the code your python code needs and reinstall them in tpython 3
cd ~/python3_ws/src
git clone https://github.com/ros/geometry.git
git clone https://github.com/ros/geometry2.git
# the dependencies of the previos package are the following
pip install pyaml
pip install rospkg
pip install empy

compile the catkin workspace
cd ~/python3_ws
compile
##error may occur
CMake Error at geometry/tf/CMakeLists.txt:87 (target_link_libraries):
  Attempt to add link library "tf" to target "test_transform_datatypes" which
  is not built in this directory.

  This is allowed only when policy CMP0079 is set to NEW.
  #fix
https://githubmemory.com/repo/ros/geometry/issues/213
catkin_add_gtest(test_transform2_datatypes test/test_transform_datatypes.cpp)
target_link_libraries(test_transform2_datatypes tf2  ${console_bridge_LIBRARIES})
add_dependencies(test_transform2_datatypes ${catkin_EXPORTED_TARGETS})


Step 4
 compile Your ros packages which depends on python3 libraries

open a new terminal with you catkin_ws- do this for evey terminal you open that require the python3
source ~/python3_ws/py3venv/bin/activate
source ~/python3_ws/devel/setup.bash

**important**
make sure to delete your devel and build folder to start clean with python 3
rm -rf /build /devel
then build as follows
catkin_make -DPYTHON_EXECUTABLE:FILEPATH=/home/stevedan/python3_ws/py3venv/bin/python
then source as always in ros
source /devel/setup.bash
Step 5  launch


**important**
in any terminal you want to use this you nned to acitvate the oyhthon evn
source ~/python3_ws/py3venv/bin/activate


TODO:
Adjust bin size
adjust dimenstion of robot based on guillium new values
define bin workspace with minomum velcoty which is 7
