#!/usr/bin/env python3

#################################################
#####    Guillem Cornella, November 2021    #####
##### The code has the following structure: #####
# · Obtain the frames from an IP Camera
# · Segment the frames to extract just the red colors
# · Apply an opening to eliminate blobs generated by a bad red segmentation
# · Extract the corners from the detected red object
# · Plot a rectangle on top of the image frames containing the red object
# · Plot the position of the centroid, and the width and height of the object
################################################
#''' Libraries '''


# rospy for the subscriber
import rospy
# ROS Image message
from sensor_msgs.msg import Image
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2
from ball_shooter_training.msg import object_tracked_info
# Instantiate CvBridge
bridge = CvBridge()

result_pub = rospy.Publisher('/object_location', object_tracked_info, queue_size=1)
# This function will generate a box containing the desired information
def __draw_label(img, text, pos, bg_color):
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
def color_segmentation(img):
    # img = cv2.imread("capture1.png")
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ## Gen lower mask (0-5) and upper mask (175-180) of RED
    mask1 = cv2.inRange(img_hsv, (0, 50, 20), (5, 255, 255))
    mask2 = cv2.inRange(img_hsv, (175, 50, 20), (180, 255, 255))
    ## Merge the mask and crop the red regions
    mask = cv2.bitwise_or(mask1, mask2)
    #cropped = cv2.bitwise_and(img, img, mask=mask)
    return mask
# This function will perform a morphological operation to eliminate blobs by Opening the binary mask
def image_opening(img):
    # Generate a morphological opening (an erosion followed by a dilation).
    kernelSize = [20,20]    # (3, 3), (5, 5), (7, 7)
    # loop over the kernels sizes
    #for kernelSize in kernelSizes:
        # construct a rectangular kernel from the current size and then
        # apply an "opening" operation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return opening
# This function will return the corners of a binary object
def get_corners(img):
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
    else:
        cnt = contours[0]
        # Coordinates of the Centroid
        M = cv2.moments(cnt)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        x, y, w, h = cv2.boundingRect(cnt)

    return x,y,w,h,cnt,cX,cY

def obtain_feature(frame):
    red_image_mask = color_segmentation(frame)
    opened_image = image_opening(red_image_mask)
    # cropped = cv2.bitwise_and(frame, frame, mask=opened_image)
    x,y,w,h,corners,cX,cY = get_corners(opened_image)
    return [w,h,cX,cY],w

def image_callback(msg):
    #("Received an image!")
    try:
        # Convert your ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
        features, object_detected = obtain_feature(cv2_img)
        data_to_send = object_tracked_info()
        data_to_send.points = features
        data_to_send.object_in_frame = object_detected
        result_pub.publish(data_to_send)

    except CvBridgeError as e:
        print(e)
    else:
        # Save your OpenCV2 image as a jpeg
        cv2.imwrite('camera_image.jpeg', cv2_img)
def main():
    rospy.init_node('image_listener')
    # Define your image topic
    image_topic = "/camera/image_raw"
    # Set up your subscriber and define its callback
    rospy.Subscriber(image_topic, Image, image_callback)
    # result_pub = rospy.Publisher('/object_location', object_tracked_info, queue_size=1)
    # Spin until ctrl + c
    rospy.spin()

if __name__ == '__main__':
    main()
