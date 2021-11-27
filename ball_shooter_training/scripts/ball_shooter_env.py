#!/usr/bin/env python
import gym
import rospy
import time
import numpy as np
import math
import copy
import random
from gym import utils, spaces
from gazebo_connection import GazeboConnection
from controllers_connection import ControllersConnection

from gym.utils import seeding
from gym.envs.registration import register

from geometry_msgs.msg import Point
from tf.transformations import euler_from_quaternion
from ball_shooter_utils_rl import BallShooterRLUtils

reg = register(
    id='BallShooterEnv-v0',
    entry_point='ball_shooter_env:BallShooterEnv',
    max_episode_steps=50, # this should be one in thoery
    )

class BallShooterEnv(gym.Env):

    def __init__(self):



        #set parameters
        self.pan_tilt_running_step = rospy.get_param("/ball_shooter/pan_tilt_running_step")
        self.launch_running_step = rospy.get_param("/ball_shooter/launch_running_step")
        self.max_xy_ = rospy.get_param("/ball_shooter/max_xy_")
        self.min_xy_ = rospy.get_param("/ball_shooter/min_xy_")

        # ball shooter object
        self.ball_shooter_object = BallShooterRLUtils()

        # stablishes connection with simulator
        self.gazebo = GazeboConnection()
        self.controllers_list = ['joint_state_controller','pan_joint_position_controller']
        self.controllers_object = ControllersConnection(namespace="ball_shooter")

        self._seed()



    def _seed(self, seed=None): #overriden function
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self,action):
        # given the action selected by the learning algorithm
        #action is a array of 2, [vel_cmd, pan_angle(deg)]

        self.gazebo.unpauseSim()
        #we rotate the robot first until we find the bin TODO: fix other code
        state = self.ball_shooter_object.get_state()

        #get robot state-
        # rotate pan tilt, specfied rotation of the disred pose with respect to the origin
        rad = np.deg2rad(action[1])
        self.ball_shooter_object.move_pan_tilt(rad)
        time.sleep(self.pan_tilt_running_step)
        #launch the ball at a specific initial speed (should be in m/s)
        self.ball_shooter_object.launch_ball(vel_cmd = action[0])
        time.sleep(self.launch_running_step)
        self.gazebo.pauseSim()
        done = self.ball_shooter_object.observation_check()
        reward = self.ball_shooter_object.get_reward_for_observation()
        info = {}
        return state, reward, done, info
    def _reset(self):
        #Pause the sim
        rospy.loginfo("Pausing SIM..")
        self.gazebo.unpauseSim()
        #resetting the simulation
        rospy.loginfo("Resetting SIM..")
        self.gazebo.resetSim()
        # Reset controller
        rospy.loginfo("Reset pan_joint_position_controller..")
        self.controllers_object.reset_ball_shooter_joint_controllers()
        #reset robot to intial position
        rospy.loginfo("Setting initial position")
        self.ball_shooter_object.set_init_pose()
        #randonly place the bin in the environment TODO: check if this is done every step it reset
        x = random.uniform(self.min_xy_, self.max_xy_)
        y = random.uniform(self.min_xy_, self.max_xy_)
        rospy.loginfo("Setting at location ==> x: " + str(x) + " y: " + str(y))
        #self.ball_shooter_object.set_bin_location(x=x,y=y)

        #check that all subscribers and Publishers work
        rospy.loginfo("check_all_systems_subscribers_publishers_services_ready...")
        #self.ball_shooter_object.check_all_sensors_ready()
        self.ball_shooter_object.check_publisher_connection()
        self.ball_shooter_object.check_all_services()

        rospy.loginfo("Pausing simulation...")
        state = self.gazebo.pauseSim()
        # get the current state
        state = self.ball_shooter_object.get_state()
        return state

#uncomment to make sure no errors
# def ball_shooter_env_systems_test():
#     rospy.init_node('ball_shooter_env_systems_test', anonymous=True, log_level=rospy.INFO)
#     ball_shooter_env_object = BallShooterEnv()
#     # ball_shooter_env_object._reset()
#
#     action = [4,1]
#     result = ball_shooter_env_object._step(action)
#     rospy.loginfo(str(result.done))
#
#
#
# if __name__ == "__main__":
#     ball_shooter_env_systems_test()
