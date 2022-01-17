import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import tensorflow as tf
import classify_image
import ColorFiltering


class RosTensorFlow():
    def __init__(self):
        
        self._session = tf.Session()
        classify_image.create_graph()
        self._cv_bridge = CvBridge()

        self._sub = rospy.Subscriber('image', Image, self.callback, queue_size=1)
        self._pub = rospy.Publisher('result', String, queue_size=1)
        self.score_threshold = rospy.get_param('~score_threshold', 0.3)
        self.use_top_k = rospy.get_param('~use_top_k', 5)

    def callback(self, image_msg):
        numBoxes=8
        Imgs=[None]*8
        cv_image = self._cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")

        objects, position=ColorFiltering.contornos(numBoxes, cv_image)

        for i in range(numBoxes):
                Imgs[i]=[i, objects[i], self._session, self.score_threshold, self.use_top_k, position[i]]


        res=tuple(map(ColorFiltering.classification,Imgs)) 
        #escribir en el comand window
        rospy.loginfo(res)
        #publicar en el nodo
        #crear otro para publicar el video
        #no se si me dejara publicar estructuras de datos
        self._pub.publish(res)

        

    def main(self):
        rospy.spin()

if __name__ == '__main__':
    classify_image.setup_args()
    #model_dir='/tmp/imagenet'
    #image_file='camera'
    #image='/cv_camera/image_raw'
    
    rospy.init_node('Color_Filtering')
    tensor = RosTensorFlow()
    tensor.main()
