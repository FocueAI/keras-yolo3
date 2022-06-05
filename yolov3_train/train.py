"""
Retrain the YOLO model for your own dataset.
"""

import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import os

use_mobilenet_flag = True
###################################想引入darknet 原版的yolov3模型 ############################################
if use_mobilenet_flag:
    # 1. 以 mobilnet 为 backbone 的训练
    from yolo3.model_Mobilenet import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
else:
    # 2. 以原始的 darknet 为 backbone 的训练 (但是还有 Ciou 作为loss函数)
    from yolo3.model import preprocess_true_boxes,yolo_body,tiny_yolo_body
    from yolo3.model_Mobilenet import yolo_loss
################################################################################################################################


from yolo3.utils import get_random_data
import tensorflow as tf

def _main():
    # train_path = '/home/aibc/MrH/project/keras-yolo3/card/2007_train.txt'
    train_path = r'./2007_train_datas.txt'
    log_dir = 'logs_mobilenet/Mobilenet/'
    log_test_dir = 'logs_test'
    classes_path = 'model_data/detect_obj_names.txt'
    anchors_path = './yolo_anchors.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1, allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    K.set_session(sess)

    input_shape = (416,416) # multiple of 32, hw

    is_tiny_version = len(anchors)==6  # default setting
    if is_tiny_version:
        model = create_tiny_model(input_shape, anchors, num_classes,
            freeze_body=2)
    else:
        model = create_model(input_shape, anchors, num_classes,load_pretrained=False,freeze_body=2) # make sure you know what you freeze

    logging = TensorBoard(log_dir=log_test_dir)
    # checkpoint = ModelCheckpoint(log_dir + 'car_mobilenet_yolov3.ckpt',
    #    monitor='val_loss', save_weights_only=False, period=1)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, min_lr=1e-9, patience=5, verbose=1)
    #early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    val_split = 0.25 #optimize:0.1 --the ratio of valdation datasets
    with open(train_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    # 增加 lr 值的 打印 2022—6-5
    def get_lr_metric(optimizer):
        def lr(y_true, y_pred):
            return optimizer.lr
        return lr

    optimizer = Adam(lr=1e-3)
    lr_metric = get_lr_metric(optimizer)

    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable= True
        # model.compile(optimizer=Adam(lr=1e-3), loss=lambda y_true, y_pred: y_pred) # raw
        model.compile(optimizer=optimizer, loss=lambda y_true, y_pred: y_pred, metrics=['accuracy',lr_metric]) # 增加 lr 值的 打印 2022—6-5
        batch_size = 1 # note that more GPU memory is required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val//batch_size),
			epochs=200,
            initial_epoch=0,
            callbacks=[logging, checkpoint, reduce_lr])

       
        model.save_weights(log_dir + 'final.h5')



def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, load_pretrained=False, freeze_body=2,
            weights_path='./model_data/predition_model/ep183-loss7.760-val_loss6.792.h5'):
    '''create the training model'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    #image_input = Input(shape=(416, 416, 3)) # 临时修改的
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)] # 等于这是一个列表，里面的元素是placehold数据类型，需要在模型训练的时候，喂入对应的具体值
    # 构建主干网络
    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        #if freeze_body in [1, 2]:
        #    # Freeze darknet53 body or freeze all but 3 output layers.
        #    num = (185, len(model_body.layers)-3)[freeze_body-1]
        #    for i in range(num): model_body.layers[i].trainable = False
        #    print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    print(model_body.output)
    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    #     输入的图片尺寸对应的tensor，y_true
    model = Model([model_body.input, *y_true], model_loss)
    model.summary()
    return model

def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/predition_model/tiny_yolo_weights.h5'):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], \
        num_anchors//2, num_classes+5)) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors//2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze the darknet body or freeze all but 2 output layers.
            num = (20, len(model_body.layers)-2)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)
    

    return model

def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines) # 当从新一轮处理数据时，就要原地打乱数据
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)

def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)

if __name__ == '__main__':
    _main()

# if __name__ == '__main__':
#     train_path = r'./2007_train_datas.txt'
#     classes_path = 'model_data/detect_obj_names.txt'
#     anchors_path = './yolo_anchors.txt'
#     class_names = get_classes(classes_path)
#     num_classes = len(class_names)
#     anchors = get_anchors(anchors_path)
#
#     with open(train_path) as f:
#         lines = f.readlines()
#     np.random.shuffle(lines)
#     gen_datasets = data_generator(annotation_lines=lines,batch_size=1,input_shape=(416,416),anchors=anchors,num_classes=num_classes)
#     for i in gen_datasets:
#         print('img:',i[0][0].shape)
#         print('label:',i[1])
#         break

