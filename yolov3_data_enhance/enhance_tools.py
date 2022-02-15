import numpy as np
import cv2
import os
from PIL import Image

save_path = './trans_pic'
raw_pic = './raw_pic'

def pil2cv(pil_img):
    ''' 把 输入的 pil格式的图像数据转换成 opencv格式的数据 '''
    return  cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)

def cv2pil(img):
    ''' 把 opecv 格式的图像 转换成 pil格式的图像 '''
    return Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))



def SaltAndPepper(src ,percetage):
    SP_NoiseImg =src.copy()
    SP_NoiseNum =int(percetage *src.shape[0] *src.shape[1])
    for i in range(SP_NoiseNum):
        randR =np.random.randint(0 ,src.shape[0 ] -1)
        randG =np.random.randint(0 ,src.shape[1 ] -1)
        randB =np.random.randint(0 ,3)
        if np.random.randint(0 ,1 )==0:
            SP_NoiseImg[randR ,randG ,randB ] =0
        else:
            SP_NoiseImg[randR ,randG ,randB ] =255
    return SP_NoiseImg

def addGaussianNoise(image ,percetage):
    G_Noiseimg = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    G_NoiseNum =int(percetage *image.shape[0 ] *image.shape[1])
    for i in range(G_NoiseNum):
        temp_x = np.random.randint(0 ,h)
        temp_y = np.random.randint(0 ,w)
        G_Noiseimg[temp_x][temp_y][np.random.randint(3)] = np.random.randn(1)[0]
    return G_Noiseimg

# dimming
def darker(image ,percetage=0.9):
    image_copy = image.copy()
    # w, h = image_copy.size
    w = image.shape[1]
    h = image.shape[0]
    # get darker
    for xi in range(0 ,w):
        for xj in range(0 ,h):
            image_copy[xj ,xi ,0] = int(image[xj ,xi ,0 ] *percetage)
            image_copy[xj ,xi ,1] = int(image[xj ,xi ,1 ] *percetage)
            image_copy[xj ,xi ,2] = int(image[xj ,xi ,2 ] *percetage)
    return image_copy

def brighter(image, percetage=1.5):
    image_copy = image.copy()
    # w, h = image_copy.size
    h = image.shape[0]
    w = image.shape[1]
    # get brighter
    for xi in range(0 ,w):
        for xj in range(0 ,h):
            image_copy[xj ,xi ,0] = np.clip(int(image[xj ,xi ,0 ] *percetage) ,a_max=255 ,a_min=0)
            image_copy[xj ,xi ,1] = np.clip(int(image[xj ,xi ,1 ] *percetage) ,a_max=255 ,a_min=0)
            image_copy[xj ,xi ,2] = np.clip(int(image[xj ,xi ,2 ] *percetage) ,a_max=255 ,a_min=0)
    return image_copy

def rotate(image, angle=15, scale=0.9):
    w = image.shape[1]
    h = image.shape[0]
    # rotate matrix
    M = cv2.getRotationMatrix2D(( w /2 , h /2), angle, scale)
    # rotate
    image = cv2.warpAffine(image ,M ,(w ,h))
    return image
def img_augmentation(path, name_int):
    img = cv2.imread(path)
    img_flip = cv2.flip(img ,1  )  # flip
    img_rotation = rotate(img  )  # rotation

    img_noise1 = SaltAndPepper(img, 0.3)
    img_noise2 = addGaussianNoise(img, 0.3)

    img_brighter = brighter(img)
    img_darker = darker(img)

    cv2.imwrite(save_path +'%s' %str(name_int) + '.jpg', img_flip)
    cv2.imwrite(save_path + '%s' % str(name_int + 1) + '.jpg', img_rotation)
    cv2.imwrite(save_path + '%s' % str(name_int + 2) + '.jpg', img_noise1)
    cv2.imwrite(save_path + '%s' % str(name_int + 3) + '.jpg', img_noise2)
    cv2.imwrite(save_path + '%s' % str(name_int + 4) + '.jpg', img_brighter)
    cv2.imwrite(save_path + '%s' % str(name_int + 5) + '.jpg', img_darker)
    print('over')
# TODO: https://blog.csdn.net/u011984148/article/details/107572526
#
# TODO： 1. 自己先实现了一般较为粗糙的随机擦除的问题
def random_cut(image):
    ''' 水机'''
    w = image.shape[1]
    h = image.shape[0]

    eraser_zone_width = random.randint(w // 10, w // 6)
    eraser_zone_height = random.randint(h // 10, h // 6)
    left_x = random.randint(0,w-eraser_zone_width)
    left_y = random.randint(0,h-eraser_zone_height)
    r = random.randint(0,255)
    g = random.randint(0,255)
    b = random.randint(0,255)
    image[left_y:eraser_zone_height,left_x:eraser_zone_width,:] = [b,g,r]

    return image
############
def color_shake(cv2_img):
    ''' 颜色抖动 '''
    image = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
    img = rgb_to_hsv(np.array(image) / 255.)
    img = hsv_to_rgb(img) * 255.
    # img = np.transpose(img,(2,1,0))
    return img

def rgb_to_hsv(arr):
    """
    convert float rgb values (in the range [0, 1]), in a numpy array to hsv
    values.

    Parameters
    ----------
    arr : (..., 3) array-like
       All values must be in the range [0, 1]

    Returns
    -------
    hsv : (..., 3) ndarray
       Colors converted to hsv values in range [0, 1]
    """
    # make sure it is an ndarray
    arr = np.asarray(arr)

    # check length of the last dimension, should be _some_ sort of rgb
    if arr.shape[-1] != 3:
        raise ValueError("Last dimension of input array must be 3; "
                         "shape {} was found.".format(arr.shape))

    in_ndim = arr.ndim
    if arr.ndim == 1:
        arr = np.array(arr, ndmin=2)

    # make sure we don't have an int image
    arr = arr.astype(np.promote_types(arr.dtype, np.float32))

    out = np.zeros_like(arr)
    arr_max = arr.max(-1)
    ipos = arr_max > 0
    delta = arr.ptp(-1)
    s = np.zeros_like(delta)
    s[ipos] = delta[ipos] / arr_max[ipos]
    ipos = delta > 0
    # red is max
    idx = (arr[..., 0] == arr_max) & ipos
    out[idx, 0] = (arr[idx, 1] - arr[idx, 2]) / delta[idx]
    # green is max
    idx = (arr[..., 1] == arr_max) & ipos
    out[idx, 0] = 2. + (arr[idx, 2] - arr[idx, 0]) / delta[idx]
    # blue is max
    idx = (arr[..., 2] == arr_max) & ipos
    out[idx, 0] = 4. + (arr[idx, 0] - arr[idx, 1]) / delta[idx]

    out[..., 0] = (out[..., 0] / 6.0) % 1.0
    out[..., 1] = s
    out[..., 2] = arr_max

    if in_ndim == 1:
        out.shape = (3,)

    return out


def hsv_to_rgb(hsv):
    """
    convert hsv values in a numpy array to rgb values
    all values assumed to be in range [0, 1]

    Parameters
    ----------
    hsv : (..., 3) array-like
       All values assumed to be in range [0, 1]

    Returns
    -------
    rgb : (..., 3) ndarray
       Colors converted to RGB values in range [0, 1]
    """
    hsv = np.asarray(hsv)

    # check length of the last dimension, should be _some_ sort of rgb
    if hsv.shape[-1] != 3:
        raise ValueError("Last dimension of input array must be 3; "
                         "shape {shp} was found.".format(shp=hsv.shape))

    # if we got passed a 1D array, try to treat as
    # a single color and reshape as needed
    in_ndim = hsv.ndim
    if in_ndim == 1:
        hsv = np.array(hsv, ndmin=2)

    # make sure we don't have an int image
    hsv = hsv.astype(np.promote_types(hsv.dtype, np.float32))

    h = hsv[..., 0]
    s = hsv[..., 1]
    v = hsv[..., 2]

    r = np.empty_like(h)
    g = np.empty_like(h)
    b = np.empty_like(h)

    i = (h * 6.0).astype(int)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    idx = i % 6 == 0
    r[idx] = v[idx]
    g[idx] = t[idx]
    b[idx] = p[idx]

    idx = i == 1
    r[idx] = q[idx]
    g[idx] = v[idx]
    b[idx] = p[idx]

    idx = i == 2
    r[idx] = p[idx]
    g[idx] = v[idx]
    b[idx] = t[idx]

    idx = i == 3
    r[idx] = p[idx]
    g[idx] = q[idx]
    b[idx] = v[idx]

    idx = i == 4
    r[idx] = t[idx]
    g[idx] = p[idx]
    b[idx] = v[idx]

    idx = i == 5
    r[idx] = v[idx]
    g[idx] = p[idx]
    b[idx] = q[idx]

    idx = s == 0
    r[idx] = v[idx]
    g[idx] = v[idx]
    b[idx] = v[idx]

    # `np.stack([r, g, b], axis=-1)` (numpy 1.10).
    rgb = np.concatenate([r[..., None], g[..., None], b[..., None]], -1)

    if in_ndim == 1:
        rgb.shape = (3,)

    return rgb

def  a():
    pass
###########################---<弹性形变>----##########################################
# 参考文档： https://zhuanlan.zhihu.com/p/46833956
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.
     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]   #(512,512)表示图像的尺寸
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    # pts1为变换前的坐标，pts2为变换后的坐标，范围为什么是center_square+-square_size？
    # 其中center_square是图像的中心，square_size=512//3=170
    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    # Mat getAffineTransform(InputArray src, InputArray dst)  src表示输入的三个点，dst表示输出的三个点，获取变换矩阵M
    M = cv2.getAffineTransform(pts1, pts2)  #获取变换矩阵
    #默认使用 双线性插值，
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    # # random_state.rand(*shape) 会产生一个和 shape 一样打的服从[0,1]均匀分布的矩阵
    # * 2 - 1 是为了将分布平移到 [-1, 1] 的区间
    # 对random_state.rand(*shape)做高斯卷积，没有对图像做高斯卷积，为什么？因为论文上这样操作的
    # 高斯卷积原理可参考：https://blog.csdn.net/sunmc1204953974/article/details/50634652
    # 实际上 dx 和 dy 就是在计算论文中弹性变换的那三步：产生一个随机的位移，将卷积核作用在上面，用 alpha 决定尺度的大小
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)  #构造一个尺寸与dx相同的O矩阵
    # np.meshgrid 生成网格点坐标矩阵，并在生成的网格点坐标矩阵上加上刚刚的到的dx dy
    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))  #网格采样点函数
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    # indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))
    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

def draw_grid(im, grid_size):
    # Draw grid lines
    for i in range(0, im.shape[1], grid_size):
        cv2.line(im, (i, 0), (i, im.shape[0]), color=(255,))
    for j in range(0, im.shape[0], grid_size):
        cv2.line(im, (0, j), (im.shape[1], j), color=(255,))

#####################################################################

from tqdm import tqdm
import shutil
if __name__ == '__main__':
    # for index,i in enumerate(os.listdir(raw_pic)):
    #     print(i)
    #     # name = i.split('.')[0]
    #     tot_path = os.path.join(raw_pic,i)
    #     img_augmentation(tot_path,index)
    # img = cv2.imread('./hou_yyzz_test.jpg')
    # print('before:',img.shape)
    # img = cv2.resize(img,(int(img.shape[1]*0.8),int(img.shape[0]*0.8)))
    # print('after:', img.shape)
    # # img = brighter(img, percetage=1.5)
    # img_noise1 = SaltAndPepper(img, 0.05)
    # cv2.imshow('hahh',img_noise1)
    # cv2.waitKey(0)

    raw_path = r'./000000.jpg'
    result = './res'
    if os.path.exists(result): shutil.rmtree(result)
    os.mkdir(result)
    for pic in tqdm(os.listdir(raw_path)):
        tail_path = raw_path + '/' + pic
        save_path = result + '/'+pic.split('.')[0]+'hengen2019'+'.jpg'
        img = cv2.imread(tail_path)
        img = cv2.resize(img, (int(img.shape[1] * 0.8), int(img.shape[0] * 0.8)))
        img_noise1 = SaltAndPepper(img, 0.05)
        cv2.imwrite(save_path,img_noise1)














