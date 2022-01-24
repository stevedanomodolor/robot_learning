import time
import rospy
import math
import copy
import numpy
import numpy as np
# from ball_shooter_training.msg import object_tracked_info
import socket
import struct

# vision import

# rospy for the subscriber
import cv2

class VisionDetection():
    def __init__(self):
        print("Trying to connect to camera...")
        self.cap = cv2.VideoCapture('http://192.168.43.1:8080/video')
        print("Connected.")
    # This function will generate a box containing the desired information
    def __draw_label(self, img, text, pos, bg_color):
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        color = (0, 0, 0)
        thickness = cv2.FILLED
        margin = 5
        txt_size = cv2.getTextSize(text, font_face, scale, thickness)
        end_x = pos[0] + txt_size[0][0] + margin
        end_y = pos[1] - txt_size[0][1] - margin

        cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
        cv2.putText(img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)
    # This function will segment the image by color and will create a mask with just the red objects
    def color_segmentation(self,img):
        print("Red color segmentation...")
        # img = cv2.imread("capture1.png")
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        ## Gen lower mask (0-5) and upper mask (175-180) of RED
        mask1 = cv2.inRange(img_hsv, (0, 50, 20), (5, 255, 255))
        mask2 = cv2.inRange(img_hsv, (175, 50, 20), (180, 255, 255))
        ## Merge the mask and crop the red regions
        mask = cv2.bitwise_or(mask1, mask2)
        #cropped = cv2.bitwise_and(img, img, mask=mask)
        return mask
    def image_opening(self,img):
        print("Opening image for visualization issues...")
        # Generate a morphological opening (an erosion followed by a dilation).
        kernelSize = (20,20)    # (3, 3), (5, 5), (7, 7)
        # loop over the kernels sizes
        #for kernelSize in kernelSizes:
            # construct a rectangular kernel from the current size and then
            # apply an "opening" operation
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        return opening
    # This function will return the corners of a binary object
    def get_corners(self,img):
        print("Getting corners...")
        contours, hierarchy = cv2.findContours(img, 1, 2)
        #print(contours)
        if not(contours):
            cnt = ([0,0])
            x = [0, 0]
            y = [0, 0]
            w = 0
            h = 0
            cX = 0
            cY = 0
            bin_in_frame = False
        else:
            cnt = contours[0]
            # Coordinates of the Centroid
            M = cv2.moments(cnt)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            x, y, w, h = cv2.boundingRect(cnt)
            bin_in_frame = True

        return x,y,w,h,cnt,cX,cY,bin_in_frame
    def obtain_feature(self):
        print("Obtaining features...")
        while(True):
            ret, frame = self.cap.read() # ret is a boolean variable that returns true if the frame is available.
            # frame is an image array vector captured based on the default frames per second defined explicitly or implicitly
            if ret == True:
                red_image_mask = self.color_segmentation(frame)
                opened_image = self.image_opening(red_image_mask)
                # cropped = cv2.bitwise_and(frame, frame, mask=opened_image)
                x,y,w,h,corners,cX,cY, bin_in_frame = self.get_corners(opened_image)
                if(bin_in_frame):
                    A = [int((cX-w/2)),int((cY-(h/2)))]
                    B = [int((cX+w/2)),int((cY-(h/2)))]
                    C = [int((cX-w/2)),int((cY+(h/2)))]
                    D = [int((cX+w/2)),int((cY+(h/2)))]
                    # [A(0), B(0), C(0), D(0), A(1), B(1), C(1), D(1)]
                    state = ([A[0], B[0], C[0], D[0], A[1], B[1], C[1], D[1]])
                    # state_int = [int(x) for x in state]
                    break
        # print(state)
        return state

# Class to interact with gazebo
class BallShooterRLUtilsRealRobot(object):
    def __init__(self):
        payload_out = bytes()
        payload_out = struct.Struct("ff").size


        #define pfixed pitch value
        # rospy.Subscriber("/object_location", object_tracked_info, self.object_location_callback)
        # # socket communcation
        # TODO
        # ESP8266AddressPort   = ("192.168.43.16", 8888)
        # bufferSize          = 1024
        # UDPSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

        self.vision_object = VisionDetection()
    def get_state(self):
        print("Obtaining object state...")
        state = self.vision_object.obtain_feature()
        print("Object state obtained")
        return state

    def move_pan_tilt_launch_ball(self, action):
        #action[0]- velocity
        #action[1] - position
        action_array = action[0]
        print("Send action through UDP socket...")
        print("Action to be sent is: " + str(action_array))

        payload_out = struct.pack("ff", action_array[0][1], action_array[0][0])
        #TODO: uncommnet when robot is connected
        # self.UDPSocket.sendto(payload_out, ESP8266AddressPort)
        #payload_in = struct.unpack("ff", payload_out)
        print("Command sent: " +   str(payload_out))

#uncomment to test functions
# def ball_shooter_rl_systems_test():
#     # rospy.init_node('ball_shooter_systems_test_node', anonymous=True, log_level=rospy.INFO)
#     ball_shooter_rl_utils_object = BallShooterRLUtilsRealRobot()
#     object_state = ball_shooter_rl_utils_object.move_pan_tilt_launch_ball(-30, 0.15)
#
#
# if __name__ == "__main__":
#     ball_shooter_rl_systems_test()
