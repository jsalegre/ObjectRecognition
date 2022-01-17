import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import tensorflow as tf
from darkflow.net.build import TFNet
import YOLO_func


class RosTensorFlow():
    def __init__(self):
        
        
        self._cv_bridge = CvBridge()

        options={
        'model':'cfg/yolo.cfg',
        'load': 'bin/yolov2.weights',
        'threshold':0.4
        }

        tfnet=TFNet(options)

        self._sub = rospy.Subscriber('image', Image, self.callback, queue_size=1)
        self._pub = rospy.Publisher('result', String, queue_size=1)
        

    def callback(self, image_msg):
        

        cv_image = self._cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")

        results=YOLO_func.classify(tfnet,cv_image)


        rospy.loginfo(results)
        #publicar en el nodo
        #crear otro para publicar el video
        #no se si me dejara publicar estructuras de datos
        self._pub.publish(results)

        

    def main(self):
        rospy.spin()

if __name__ == '__main__':


    
    #model_dir='/tmp/imagenet'
    #image_file='camera'
    #image='/cv_camera/image_raw'
    
    rospy.init_node('YOLO')
    tensor = RosTensorFlow()
    tensor.main()
