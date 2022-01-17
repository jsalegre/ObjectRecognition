import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import tensorflow as tf
from darkflow.net.build import TFNet
import SSD_func
sys.path.append("..")
from object_detection.utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util


class RosTensorFlow():
    def __init__(self):
        
        
        self._cv_bridge = CvBridge()

        self.MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
        self.PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
        self.PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
        self.NUM_CLASSES = 90

        self.detection_graph = tf.Graph()

        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')


        self.label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        self.categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

        

        self._sub = rospy.Subscriber('image', Image, self.callback, queue_size=1)
        self._pub = rospy.Publisher('result', String, queue_size=1)
        

    def callback(self, image_msg):
        

        image_np = self._cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        output_dict = run_inference_for_single_image(image_np, detection_graph)


        # Visualization of the results of a detection.
        #vis_util.visualize_boxes_and_labels_on_image_array(
        #image_np,
        #output_dict['detection_boxes'],
        #output_dict['detection_classes'],
        #output_dict['detection_scores'],
        #category_index,
        #instance_masks=output_dict.get('detection_masks'),
        #use_normalized_coordinates=True,
        #line_thickness=8)



        rospy.loginfo(output_dict)
        #publicar en el nodo
        #crear otro para publicar el video
        #no se si me dejara publicar estructuras de datos
        self._pub.publish(output_dict)

        

    def main(self):
        rospy.spin()

if __name__ == '__main__':
    
    #model_dir='/tmp/imagenet'
    #image_file='camera'
    #image='/cv_camera/image_raw'
    
    rospy.init_node('SSD')
    tensor = RosTensorFlow()
    tensor.main()
