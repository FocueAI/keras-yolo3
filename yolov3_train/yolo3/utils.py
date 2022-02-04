from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from functools import reduce

from PIL import Image
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')

def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

def get_random_data(annotation_line, input_shape, random=True, max_boxes=20, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True):
    '''random preprocessing for real-time data augmentation'''
    line = annotation_line.split()
    image = Image.open(line[0]) # 因为是采用PIL库打开图像，所以图像的类型并不是 numpy类型的
    iw, ih = image.size
    h, w = input_shape
    box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])# box的位置与大小都采用 整形数据处理的

    if not random:  # 和预测处理数据的逻辑一致
        # resize image
        scale = min(w/iw, h/ih)  # w是输入到模型的固定的宽， iw是图片真正的宽度， 找到较小的形变因子
        nw = int(iw*scale)       # 新图像的 宽
        nh = int(ih*scale)       # 新图像的 高
        dx = (w-nw)//2
        dy = (h-nh)//2
        image_data=0
        if proc_img:
            image = image.resize((nw,nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)/255.

        # correct boxes
        box_data = np.zeros((max_boxes,5))
        if len(box)>0:
            np.random.shuffle(box)
            if len(box)>max_boxes: box = box[:max_boxes]
            box[:, [0,2]] = box[:, [0,2]]*scale + dx
            box[:, [1,3]] = box[:, [1,3]]*scale + dy
            box_data[:len(box)] = box

        return image_data, box_data

    # resize image
    new_ar = w/h * rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter)
    scale = rand(.25, 2)
    if new_ar < 1:
        nh = int(scale*h)
        nw = int(nh*new_ar)
    else:
        nw = int(scale*w)
        nh = int(nw/new_ar)
    image = image.resize((nw,nh), Image.BICUBIC)

    # place image
    dx = int(rand(0, w-nw))          # 这种贴图 左上角位置是随机的，也等于增强数据的概念
    dy = int(rand(0, h-nh))          # 这样做的目的是 防止把图片 粘贴 “出界”
    new_image = Image.new('RGB', (w,h), (128,128,128))
    new_image.paste(image, (dx, dy)) # 指定了要粘贴的左上角坐标
    image = new_image

    # flip image or not
    flip = rand()<.5
    if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)  # 左右反转

    # distort image
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
    val = rand(1, val) if rand()<.5 else 1/rand(1, val)
    x = rgb_to_hsv(np.array(image)/255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0]>1] -= 1
    x[..., 0][x[..., 0]<0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x>1] = 1
    x[x<0] = 0
    image_data = hsv_to_rgb(x) # numpy array, 0 to 1  ====》已经变成了numpy 类型

    # correct boxes
    box_data = np.zeros((max_boxes,5)) # 数据格式 [[x0,y0,x1,y1,cls],[...], ..... ]
    if len(box)>0:
        np.random.shuffle(box)
        box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx   # 根据图像变化后 的box的 x0 和 x1
        box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy   # 根据图像变化后 的box的 y0 和 y1
        if flip: box[:, [0,2]] = w - box[:, [2,0]] # 如果 图像左右反转
        box[:, 0:2][box[:, 0:2]<0] = 0             # 如果 左上角坐标 < 0 ,则给它 赋值为 0
        box[:, 2][box[:, 2]>w] = w                 # 右下角 的x 若> w, 则 给它赋值w
        box[:, 3][box[:, 3]>h] = h                 # 右下角 的y 若> h, 则 给它赋值h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box 为什么不是判断为 > 0 , ||  > 1 有什么特殊含义？？？？？？ ||
        if len(box)>max_boxes: box = box[:max_boxes]  # 因为 box 都采用整形 数据
        box_data[:len(box)] = box

    return image_data, box_data

