#!/usr/bin/env python3

import time
import rospy
import math
import copy
import numpy
from std_msgs.msg import Float64
from std_msgs.msg import Bool
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from geometry_msgs.msg import Twist
from tf.transformations import euler_from_quaternion
from ball_shooter_training.msg import object_tracked_info
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState


class BallShooterRLUtils(object):
    def __init__(self):
        #define pfixed pitch value
        self.pitch = 0.785398
        rospy.Subscriber("ball_shooter/joint_states", JointState, self.ball_shooter_joints_callback)
        #sunscribers to the bin/ball/ball_shooter odom
        rospy.Subscriber("/ball/odom", Odometry, self.ball_odom_callback)
        rospy.Subscriber("/bin/odom", Odometry, self.bin_odom_callback)
        rospy.Subscriber("/ball_shooter/odom", Odometry, self.ball_shooter_odom_callback)
        rospy.Subscriber("/object_location", object_tracked_info, self.object_location_callback)
        #publish command value for pan
        self._ball_shooter_pan_publisher = rospy.Publisher("/ball_shooter/pan_joint_position_controller/command",Float64, queue_size=1)
        # set ball velocity pub
        self.set_ball_vel_cmd = rospy.Publisher("/ball/vel_cmd",Twist, queue_size=1)
        self.activate_launch_pub = rospy.Publisher("/ball/activate_launch",Bool, queue_size=1)
        #waiting for service to be available
        self.check_all_services()
        self.check_all_sensors_ready() #TODO fix when alrady active
        self.check_publisher_connection()

    #check all sensor ready
    def check_all_sensors_ready(self):
        #pan joint
        self.ball_shooter_joint = None
        while self.ball_shooter_joint is None and not rospy.is_shutdown():
            try:
                self.ball_shooter_joint = rospy.wait_for_message("/ball_shooter/joint_states", JointState, timeout==1.0)
                rospy.loginfo("Current /ball_shooter/joint_states READY ==>" + str(self.ball_shooter_joint))
            except:
                rospy.logerr("Current Current /ball_shooter/joint_states not ready yet, retrying for getting joints")
        # Ball odom
        self.ball_odom = None
        while self.ball_odom is None and not rospy.is_shutdown():
            try:
                self.ball_odom = rospy.wait_for_message("/ball/odom", JointState, timeout==1.0)
                rospy.loginfo("Current /ball/odom READY ==>" + str(self.ball_odom))
            except:
                rospy.logerr("Current Current /ball/odom not ready yet, retrying for getting joints")
        #bin odom
        self.bin_odom = None
        while self.bin_odom is None and not rospy.is_shutdown():
            try:
                self.bin_odom = rospy.wait_for_message("/bin/odom", JointState, timeout==1.0)
                rospy.loginfo("Current /bin/odom READY ==>" + str(self.bin_odom))
            except:
                rospy.logerr("Current Current /bin/odom not ready yet, retrying for getting joints")
        #TODO add
        self.object_info = object_tracked_info()
        self.object_state = None

        rospy.loginfo("ALL SENSORS READY")

    def check_publisher_connection(self):
        rate = rospy.Rate(10) #10hz
        while(self._ball_shooter_pan_publisher.get_num_connections() == 0 and not rospy.is_shutdown()):
            rospy.loginfo("No subscribers to _ball_shooter_pan_publisher yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException():
                #this is to avoid error when the world is rested, time when backwards
                pass
                rospy.logerr("set_ball_vel_cmd Publisher Connected")
        while(self.set_ball_vel_cmd.get_num_connections() == 0 and not rospy.is_shutdown()):
            rospy.loginfo("No subscribers to set_ball_vel_cmd yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException():
                #this is to avoid error when the world is rested, time when backwards
                pass
                rospy.logerr("set_ball_vel_cmd Publisher Connected")
        while(self.activate_launch_pub.get_num_connections() == 0 and not rospy.is_shutdown()):
            rospy.loginfo("No subscribers to activate_launch_pub yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException():
                #this is to avoid error when the world is rested, time when backwards
                pass
                rospy.logerr("activate_launch_pub Publisher Connected")
        rospy.loginfo("All Publishers READY")

    #callbacks
    def ball_shooter_joints_callback(self, data):
        self.ball_shooter_joint = data
    def ball_odom_callback(self, data):
        self.ball_odom = data
    def bin_odom_callback(self, data):
        self.bin_odom = data
    def ball_shooter_odom_callback(self, data):
        self.ball_shooter_odom = data
    def object_location_callback(self, data):
        self.object_info = data
    def get_pan_joint(self):
        return self.ball_shooter_joint.position[0]
    def get_bin_pose(self):
        return self.bin_odom.pose.pose
    # actions
    def move_pan_tilt(self, position):
        joint_position = Float64()
        joint_position.data = position
        # rospy.loginfo("Moving pan tilt to joint >>" + str(joint_position))
        self._ball_shooter_pan_publisher.publish(joint_position)
    def launch_ball(self, vel_cmd):
        ball_linear_velocity = Twist()
        zAdjust = math.cos(self.pitch)
        yaw = self.get_pan_joint()
        ball_linear_velocity.linear.x = vel_cmd*(math.cos(yaw)*zAdjust)
        ball_linear_velocity.linear.y = vel_cmd*(math.sin(yaw)*zAdjust)
        ball_linear_velocity.linear.z = vel_cmd*math.sin(self.pitch)
        rospy.loginfo(str(ball_linear_velocity))
        self.set_ball_vel_cmd.publish(ball_linear_velocity)
        dummy_boolean = Bool()
        dummy_boolean.data = True
        self.activate_launch_pub.publish(dummy_boolean)
    def set_init_pose(self):
        self.check_publisher_connection
        self.move_pan_tilt(0)

    def move_pan_to_view_bin(self):
        #state is the for points of the bin TODO: change
        while(0):#not object_info.object_in_frame):
            joint_position_value = self.ball_shooter_joint[0] #pan joint value
            rotation_direction = 1 # rotation direction
            #extrem
            if((joint_position_value+0.05)> 3.14):
                #rotate the other direction
                #joint_position_value.data= joint_position_value.data-0.05;
                rotation_direction = -1
            elif((joint_position_value-0.05)<-3.14):
                #rotate in the other direction
                #joint_position_value.data= joint_position_value.data+0.05;
                rotation_direction = 1
            joint_position_value = joint_position_value+(0.05*rotation_direction)
            #move the pan until the bin is detected
            time.sleep(2)
            self.move_pan_tilt(joint_position_value)
        return self.get_object_state()

    def bin_in_view(self):
        if(self.object_info.object_in_frame):
            return True
        else:
            return False
    def get_object_state(self):
        #TODO remove
        return [1,1,1,1] #self.object_info.points

    def get_state(self):
        if(0): #not self.bin_in_view()): # TODO fix
            object_state = self.move_pan_to_view_bin()
        else:
            object_state = self.get_object_state()
        return object_state

    def check_all_services(self):
        rospy.loginfo("Resetting /gazebo/set_model_state server")
        rospy.wait_for_service("/gazebo/set_model_state")
        rospy.loginfo("All server Ready")

    def set_bin_location(self, x,y):
        action_completed = False
        bin_state_msg = ModelState()
        bin_state_msg.model_name = 'bin'
        bin_state_msg.pose = self.get_bin_pose()
        bin_state_msg.pose.position.x = x
        bin_state_msg.pose.position.y = y
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state( bin_state_msg )
            action_completed = True

        except rospy.ServiceException as e:
            print("Service call failed: " +str(e))
            action_completed = False
        return action_completed

    def observation_check(self):
        height = self.ball_odom.pose.pose.position.z
        if height < 0.06:
            done = True
        else:
            done = False
        return done
    def get_reward_for_observation(self):
        #check if the ball is inside the area of the bin base
        radius = (0.83941+0.03)-0.7166
        dx = (self.ball_odom.pose.pose.position.x-self.bin_odom.pose.pose.position.x)
        dy = (self.ball_odom.pose.pose.position.y-self.bin_odom.pose.pose.position.y)
        square_dist = dx**2 + dy**2
        inside = square_dist <= radius ** 2
        #reward is if the ball is inside but how far it is from the center of the bin base are
        #Todo change
        if(inside):
            reward = 100-(radius*100)
        else:
            reward = -100
        return reward




#uncomment to test functions
# def ball_shooter_rl_systems_test():
#     rospy.init_node('ball_shooter_systems_test_node', anonymous=True, log_level=rospy.INFO)
#     ball_shooter_rl_utils_object = BallShooterRLUtils()
#     object_state = ball_shooter_rl_utils_object.get_state()
#
#     #get object state
#     rospy.loginfo("Object state==>"+str(object_state))
#     #move pan tilt test
#     current_pose = ball_shooter_rl_utils_object.get_pan_joint()
#     rospy.loginfo("Moving to joint angle==> " +  str(current_pose+0.5))
#     ball_shooter_rl_utils_object.move_pan_tilt(current_pose+0.5)
#     time.sleep(3)
#     rospy.loginfo("Moving to joint angle==> " +  str(current_pose-0.5))
#     ball_shooter_rl_utils_object.move_pan_tilt(current_pose-0.8)
#     time.sleep(3)
#     launch_speed = 2
#     rospy.loginfo("Launch ball at Speed ==>" + str(launch_speed))
#     ball_shooter_rl_utils_object.launch_ball(vel_cmd=launch_speed)
#     rospy.loginfo("Setting pan joint to initial position")
#     ball_shooter_rl_utils_object.set_init_pose()
#     rospy.loginfo("Testing bin set location function")
#     ball_shooter_rl_utils_object.set_bin_location(x =3,y =0)
#     done = ball_shooter_rl_utils_object.observation_check()
#     reward = ball_shooter_rl_utils_object.get_reward_for_observation()
#     rospy.loginfo("Done==>"+str(done))
#     rospy.loginfo("Reward==>"+str(reward))
#
#
# if __name__ == "__main__":
    #ball_shooter_rl_systems_test()
