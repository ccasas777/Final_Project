from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import cv2
import rospy
import datetime

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier

        PATH = "light_classification/frozen_inference_graph.pb"
        self.graph = tf.Graph()
        self.threshold = 0.8

        with self.graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH, 'rb') as fid:
                od_graph_def.ParseFromString(fid.read())
                tf.import_graph_def(od_graph_def, name='')

            self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
            self.boxes = self.graph.get_tensor_by_name('detection_boxes:0')
            self.scores = self.graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.graph.get_tensor_by_name(
                'num_detections:0')

        self.sess = tf.Session(graph=self.graph)


    def filter_box(self,min_score,boxes,scores,classes):
	n = len(classes)
	idxs=[]
	for i in range(n):
	    if scores[i]>min_score:
		idxs.append(i)
	filtered_boxes = boxes[idxs,...]
	filtered_scores = scores[idxs,...]
	filtered_classes = classes[idxs,...]
	return filtered_boxes, filtered_scores, filtered_classes

    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        with self.graph.as_default():
            img_expand = np.expand_dims(image, axis=0)
            start = datetime.datetime.now()
            (boxes, scores, classes, num_detections) = self.sess.run(
                [self.boxes, self.scores, self.classes, self.num_detections],
                feed_dict={self.image_tensor: img_expand})
            end = datetime.datetime.now()
            c = end - start
            print("time_cost: ", c.total_seconds())

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)
	
	boxes, scores, classes = self.filter_box(0.4, boxes, scores, classes)
	
        if len(scores)>0:
            if classes[0] == 3:
                return TrafficLight.GREEN
            elif classes[0] == 1:
                return TrafficLight.RED
            elif classes[0] == 2:
                return TrafficLight.YELLOW
        return TrafficLight.UNKNOWN

