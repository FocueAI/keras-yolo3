#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run a YOLO_v3 style detection model on test images.
"""

import colorsys
import os
from timeit import default_timer as timer
import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
import keras
from PIL import Image, ImageFont, ImageDraw

from yolo3.model_Mobilenet import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from keras.utils import multi_gpu_model
gpu_num=1

class YOLO(object):
    def __init__(self,use_pb=False):
        self.use_pb = use_pb
        if use_pb:
            keras.backend.set_learning_phase(0)
            # self.model_path =  'logs_mobilenet/Mobilenet/card_mobileNet.h5'
            # self.model_path =   'logs_mobilenet/Mobilenet/ep081-loss24.951-val_loss16.089.h5'
            self.model_path = './model_data/food_loc_2021_12_20.pb'
            self.anchors_path = 'logs_mobilenet/yolo_anchors.txt'
            self.classes_path = 'logs_mobilenet/voc_classes.txt'
            #self.score = 0.3
            self.score = 0.3
            self.iou = 0.45
            self.class_names = self._get_class()
            self.anchors = self._get_anchors()
            # self.sess = K.get_session()
            self.model_image_size = (416, 416) # fixed size or (None, None), hw
            # self.boxes, self.scores, self.classes = self.generate_raw()
            self.boxes, self.scores, self.classes = self.generate()
        else:
            # self.model_path = 'logs_mobilenet/Mobilenet/yolo_food_loc.h5'
            self.model_path = r'logs_mobilenet/Mobilenet/test.h5'
            # self.model_path = r'logs_mobilenet/Mobilenet/ep183-loss7.760-val_loss6.792.h5'
            self.anchors_path = 'logs_mobilenet/yolo_anchors.txt'
            self.classes_path = 'logs_mobilenet/voc_classes.txt'
            # self.score = 0.3
            self.score = 0.1
            self.iou = 0.45
            self.class_names = self._get_class()
            self.anchors = self._get_anchors()
            self.sess = K.get_session()
            self.model_image_size = (416, 416)  # fixed size or (None, None), hw
            self.boxes, self.scores, self.classes = self.generate_raw()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)
    #######################################
    def load_graph(self, pbpath):
        with tf.gfile.GFile(pbpath, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')
        return graph
    #######################################
    def generate(self,pbpath='./model_data/card_mobileNet.pb'):
        '''to generate the bounding boxes'''
        self.model_path = pbpath
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.pb'), 'Keras model or weights must be a .pb file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
              '''
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
            '''
              self.graph = self.load_graph(pbpath)
              with self.graph.as_default():
                  self.sess = tf.Session(graph=self.graph)
                  # 输入
                  self.input = self.sess.graph.get_tensor_by_name('input_1:0')

                  # 输出
                  self.boxes = self.sess.graph.get_tensor_by_name('conv2d_7/BiasAdd:0')
                  self.scores = self.sess.graph.get_tensor_by_name('conv2d_15/BiasAdd:0')
                  self.classes = self.sess.graph.get_tensor_by_name('conv2d_23/BiasAdd:0')

                  self.merge_out = [self.boxes,self.scores,self.classes]

                  # Generate colors for drawing bounding boxes.
                  hsv_tuples = [(x / len(self.class_names), 1., 1.)
                                  for x in range(len(self.class_names))]
                  self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
                  self.colors = list(
                        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                            self.colors))
                  np.random.seed(10101)  # Fixed seed for consistent colors across runs.
                  np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
                  np.random.seed(None)  # Reset seed to default.

                  # Generate output tensor targets for filtered bounding boxes.
                  self.input_image_shape = K.placeholder(shape=(2, ))
                  # if gpu_num>=2:
                  #     self.yolo_model = multi_gpu_model(self.yolo_model, gpus=gpu_num)
                  boxes, scores, classes = yolo_eval(self.merge_out, self.anchors,
                            len(self.class_names), self.input_image_shape,
                            score_threshold=self.score, iou_threshold=self.iou)

                  return boxes, scores, classes


    def generate_raw(self):
        '''to generate the bounding boxes'''
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        # default arg
        # self.yolo_model->'model_data/yolo.h5'
        # self.anchors->'model_data/yolo_anchors.txt'-> 9 scales for anchors
        return boxes, scores, classes
    def detect_image(self, img_path,img = None, single_image=True, output=list()):
        start = timer()
        # rects = []
        if img == None:
            image = Image.open(img_path)
        else:
            image = img
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        # print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        # tf.Session.run(fetches, feed_dict=None)
        # Runs the operations and evaluates the tensors in fetches.
        #
        # Args:
        # fetches: A single graph element, or a list of graph elements(described above).
        #
        # feed_dict: A dictionary that maps graph elements to values(described above).
        #
        # Returns:Either a single value if fetches is a single graph element, or a
        # list of values if fetches is a list(described above).
        if self.use_pb:
            out_boxes, out_scores, out_classes = self.sess.run(
                [self.boxes, self.scores, self.classes],
                feed_dict={
                    # self.yolo_model.input: image_data,
                    self.input: image_data,
                    self.input_image_shape: [image.size[1], image.size[0]],
                    # K.learning_phase(): 0
                })
        else:
            out_boxes, out_scores, out_classes = self.sess.run(
                [self.boxes, self.scores, self.classes],
                feed_dict={
                    self.yolo_model.input: image_data,
                    self.input_image_shape: [image.size[1], image.size[0]],
                    # K.learning_phase(): 0
                })



        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype( font='font/FiraMono-Medium.otf',                                             # font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))

        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            # label = '{} {:.2f}'.format(score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            y1, x1, y2, x2 = box
            y1 = max(0, np.floor(y1 + 0.5).astype('float32'))
            x1 = max(0, np.floor(x1 + 0.5).astype('float32'))
            y2 = min(image.size[1], np.floor(y2 + 0.5).astype('float32'))
            x2 = min(image.size[0], np.floor(x2 + 0.5).astype('float32'))
            
            #image = image.crop((x1, y1, x2, y2))
            
            print(label, (x1, y1), (x2, y2))
            bbox = dict([("score",str(score)),("x1",str(x1)),("y1", str(y1)),("x2", str(x2)),("y2", str(y2))])
            # rects.append(bbox)
            # output.append([str(x1),str(y1),str(x2),str(y2)])
            output.append([x1, y1, x2, y2])
            print('-----*-*-*-*-----output:',output)
            if y1 - label_size[1] >= 0:      # hou
                #text_origin = np.array([x1, y1 - label_size[1]])
                text_origin = np.array([x2, y1 - label_size[1]])
            else:
                #text_origin = np.array([x1, y1 + 1])
                text_origin = np.array([x2, y1 + 1])
        
            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [x1 + i, y1 + i, x2 - i, y2 - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            # draw.text(text_origin, label, fill=(0, 0, 0),font)
            del draw
        #
        end = timer()
        print(str(end - start))
        return image

        ##############################new add ###########################################
    def detect_img(self, img_path):
        dete_image = Image.open(img_path)
        detections = []
        YOLO.detect_image(image=dete_image, single_image=False, output=detections)
        return detections

    ##########################################################################
    def close_session(self):
        self.sess.close()
    def convert_model_to_pb(self, save_pb_file):
        def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
            """
            Freezes the state of a session into a prunned computation graph.

            Creates a new computation graph where variable nodes are replaced by
            constants taking their current value in the session. The new graph will be
            prunned so subgraphs that are not neccesary to compute the requested
            outputs are removed.
            @param session The TensorFlow session to be frozen.
            @param keep_var_names A list of variable names that should not be frozen,
                                  or None to freeze all the variables in the graph.
            @param output_names Names of the relevant graph outputs.
            @param clear_devices Remove the device directives from the graph for better portability.
            @return The frozen graph definition.
            """
            from tensorflow.python.framework.graph_util import convert_variables_to_constants
            graph = session.graph
            with graph.as_default():
                freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
                output_names = output_names or []
                output_names += [v.op.name for v in tf.global_variables()]
                input_graph_def = graph.as_graph_def()
                if clear_devices:
                    for node in input_graph_def.node:
                        node.device = ""
                        print(node.name)
                frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                              output_names, freeze_var_names)
                return frozen_graph

        sess = K.get_session()

        model = self.yolo_model
        output_names = [output.op.name for output in model.outputs]
        print('outputs=%s' % (output_names))

        frozen_graph = freeze_session(sess, output_names=output_names)

        from tensorflow.python.framework import graph_io

        folder, name = os.path.split(save_pb_file)
        graph_io.write_graph(frozen_graph, folder, name, as_text=False)

        print('saved the constant graph (ready for inference) at: %s' % save_pb_file)






