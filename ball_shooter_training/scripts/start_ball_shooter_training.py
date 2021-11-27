#!/usr/bin/env python
import qlearn
from gym import wrappers

#Ros packages required
import rospy
import rospkg

#inport our training environment
import ball_shooter_env

if __name__ == '__main__':

    rospy.init_node('ballShooter_gym', anonymous=True, log_level=rospy.WARN)
    #create the gym environment
    env = gym.make("BallShooterEnv-v0")
    rospkg.loginfo("Gym environment done")
    #setting the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('ball_shooter_training')
