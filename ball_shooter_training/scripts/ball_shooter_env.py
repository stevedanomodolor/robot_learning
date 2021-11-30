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
        self.n_actions = rospy.get_param("/ball_shooter/n_actions")
        self.low_vel_cmd = rospy.get_param("/ball_shooter/low_vel_cmd") #minimum speed
        self.high_vel_cmd = rospy.get_param("/ball_shooter/high_vel_cmd") #minimum speed
        self.low_incre_cmd = rospy.get_param("/ball_shooter/low_incre_cmd") #min turn
        self.high_incre_cmd = rospy.get_param("/ball_shooter/high_incre_cmd") #max turn
        self.n_incre_angle = rospy.get_param("/ball_shooter/n_incre_angle") #max turn
        self.n_incre_vel = rospy.get_param("/ball_shooter/n_incre_vel") #max turn

        #action_space
        # self.action_space = spaces.Box(
        #     np.array([self.low_vel_cmd,self.low_incre_cmd]).astype(np.float32),
        #     np.array([self.high_vel_cmd,self.high_incre_cmd]).astype(np.float32),
        # )# vel_cmd increment

        self.action_space = spaces.Discrete(self.n_incre_angle*self.n_incre_vel)


                # ball shooter object
        self.ball_shooter_object = BallShooterRLUtils()

        # stablishes connection with simulator
        self.gazebo = GazeboConnection()
        self.controllers_list = ['joint_state_controller','pan_joint_position_controller']
        self.controllers_object = ControllersConnection(namespace="ball_shooter")
        self.seed()




    def seed(self, seed=None): #overriden function
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self,action):
        # given the action selected by the learning algorithm
        #action is a array of 2, [vel_cmd, pan_angle(deg)]

        self.gazebo.unpauseSim()
        time.sleep(2)


        #we rotate the robot first until we find the bin TODO: fix other code
        state = self.ball_shooter_object.get_state()
        current_action = self.convert_action_space_2_robot_action(action)

        #get robot state-
        # rotate pan tilt, specfied rotation of the disred pose with respect to the origin
        #important change this in the real robot
        rad = np.deg2rad(current_action[1])
        # current_pan_joint = self.ball_shooter_object.get_pan_joint()
        # if((current_pan_joint+rad) > 3.14):
        #     pan_command = (current_pan_joint+rad) - 3.14
        #     pan_command_ = -3.14 + pan_command
        # elif((current_pan_joint+rad) < -3.14):
        #     pan_command = (current_pan_joint+rad) + 3.14
        #     pan_command_ = 3.14 + pan_command
        # else:
        #     pan_command_ = current_pan_joint+rad


        self.ball_shooter_object.move_pan_tilt(rad)
        time.sleep(self.pan_tilt_running_step)
        #launch the ball at a specific initial speed (should be in m/s)
        self.ball_shooter_object.launch_ball(vel_cmd = current_action[0])
        time.sleep(self.launch_running_step)
        self.gazebo.pauseSim()
        done = self.ball_shooter_object.observation_check()
        reward = self.ball_shooter_object.get_reward_for_observation()
        info = {}
        return state, reward, done, info
    def reset(self):
        #Pause the sim
        rospy.loginfo("Pausing SIM..")
        self.gazebo.unpauseSim()
        time.sleep(2)
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


    def convert_action_space_2_robot_action(self, action_space):
        action = [0, 0] # vel_cmd pan
        if action_space == 0:
            action[0] = 8
            action[1] =  0
        elif action_space == 1:
            action[0] = 8
            action[1] =-30
        elif action_space == 2:
            action[0] = 8
            action[1] =-20
        elif action_space == 3:
            action[0] = 8
            action[1] =-10
        elif action_space == 4:
            action[0] = 8
            action[1] = 10
        elif action_space == 5:
            action[0] = 8
            action[1] = 20
        elif action_space == 6:
            action[0] = 8
            action[1] = 30
        elif action_space == 7:
            action[0] = 12
            action[1] =  0
        elif action_space == 8:
            action[0] = 12
            action[1] =-30
        elif action_space == 9:
            action[0] = 12
            action[1] =-20
        elif action_space == 10:
            action[0] = 12
            action[1] =-10
        elif action_space == 11:
            action[0] = 12
            action[1] = 10
        elif action_space == 12:
            action[0] = 12
            action[1] = 20
        elif action_space == 13:
            action[0] = 12
            action[1] = 30
        elif action_space == 14:
            action[0] =16
            action[1] =  0
        elif action_space == 15:
            action[0] =16
            action[1] =-30
        elif action_space == 16:
            action[0] =16
            action[1] =-20
        elif action_space == 17:
            action[0] =16
            action[1] =-10
        elif action_space == 18:
            action[0] =16
            action[1] = 10
        elif action_space == 19:
            action[0] =16
            action[1] = 20
        elif action_space == 20:
            action[0] =16
            action[1] = 30
        elif action_space == 21:
            action[0] =20
            action[1] =  0
        elif action_space == 22:
            action[0] =20
            action[1] =-30
        elif action_space == 23:
            action[0] =20
            action[1] =-20
        elif action_space == 24:
            action[0] =20
            action[1] =-10
        elif action_space == 25:
            action[0] =20
            action[1] = 10
        elif action_space == 26:
            action[0] =20
            action[1] = 20
        elif action_space == 27:
            action[0] =20
            action[1] = 30
        return action


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
