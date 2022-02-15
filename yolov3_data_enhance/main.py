import os
import cv2
import numpy as np
import shutil
import xml.etree.ElementTree as ET
from tqdm import tqdm
import numpy as mp
import enhance_tools as tool
import random
from PIL import Image

# 原始的  图片 和 xml 文件
pic_path = r'F:\各种公司\jst2\第2批数据\all_img\traindataset\img'
label_xml_path = r'F:\各种公司\jst2\第2批数据\all_img\traindataset\label'
# 变换后的 图片 和 xml 文件
save_trans_pic_path = r'F:\各种公司\jst2\第2批数据\all_img\traindataset\img-enhance'
save_trans_xml_path = r'F:\各种公司\jst2\第2批数据\all_img\traindataset\label-enhance'

if os.path.exists(save_trans_pic_path):shutil.rmtree(save_trans_pic_path)
os.mkdir(save_trans_pic_path)

if os.path.exists(save_trans_xml_path):shutil.rmtree(save_trans_xml_path)
os.mkdir(save_trans_xml_path)

# 在图片的四周随机增加 指定增加灰边
ramdom_add_padding_flag = True
# 是否需要在图片进行仿射变换
ramdom_perspectiveTransform_flag = True
# 增强的轮次
enhance_epoch = 10

# for j in range(enhance_epoch): # 样本数要增强的倍数（不带原样本）
for j in range(0,enhance_epoch):  # 样本数要增强的倍数（不带原样本）
    for i in tqdm(os.listdir(pic_path)):
        #######################################
        random_border = random.randint(8, 20)
        random_padding_border = 60  # 随机增加padding的强度
        #######################################

        file_name,typename = os.path.splitext(i)
        detail_pic_path = os.path.join(pic_path,i)
        detail_pic_trans_path = os.path.join(save_trans_pic_path, file_name + f'_trans_{j}{typename}')
        ###########################
        # <=======before
        # img = cv2.imread(detail_pic_path)
        # =========> now
        pil_img = Image.open(detail_pic_path)
        img = tool.pil2cv(pil_img)
        ###########################

        if ramdom_add_padding_flag:  # 是否允许在图片上随机padding
            # top, bottom, left, right
            top_padding = np.random.randint(0,random_padding_border)
            bottom_padding = np.random.randint(0, random_padding_border)
            left_padding = np.random.randint(0, random_padding_border)
            right_padding = np.random.randint(0, random_padding_border)
            #                        原图      上             下            左            右               固定值-常量          常量的值
            img = cv2.copyMakeBorder(img, top_padding, bottom_padding, left_padding, right_padding, cv2.BORDER_CONSTANT,value=[128,128,128])
        img_h, img_w, _ = img.shape
        ######################################## 构建透射图，并计算透射变换矩阵---begin

        pts1 = np.float32([[0, 0],     # 原图透射前的 4个定点坐标
                           [img_w, 0],
                           [img_w, img_h],
                           [0, img_h]
                           ])
        pts2 = np.float32([[np.random.randint(-random_border,random_border), np.random.randint(-random_border,random_border)],     # 将原图上的 4个定点 透射到目标点上 ********* 在这里需要经常的变换
                           [img_w+np.random.randint(-random_border,random_border), np.random.randint(-random_border,random_border)],
                           [img_w+np.random.randint(-random_border,random_border), img_h+np.random.randint(-random_border,random_border) ],
                           [np.random.randint(-random_border,random_border), img_h+np.random.randint(-random_border,random_border)]
                           ])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst_pic = cv2.warpPerspective(img, M, (int(img_w), int(img_h)))
        ########################################## 进行数据增强操作 begin
        dst_pic = tool.addGaussianNoise(dst_pic,random.uniform(0.01, 0.03))
        dst_pic = tool.SaltAndPepper(dst_pic, random.uniform(0.01, 0.03))
        ########################################## 进行数据增强操作 end
        #<==========================before
        # cv2.imwrite(detail_pic_trans_path, dst_pic)
        # ==========================>now
        dst_pil_img = tool.cv2pil(dst_pic)
        dst_pil_img.save(detail_pic_trans_path)

        ######################################## end
        detail_xml_path = os.path.join(label_xml_path, file_name + '.xml')
        detail_xml_save_path = os.path.join(save_trans_xml_path, file_name + '_trans_%s.xml'%(j))   #######################################
        ################## 直接更新 xml 文件
        with open(detail_xml_save_path,'w',encoding='utf-8') as xml_writer:
            xml_writer.write('<annotation>\n')
            xml_writer.write('\t<folder>h</folder>\n')
            xml_writer.write('\t<filename>s</filename>\n')
            xml_writer.write('\t<path>c</path>\n')
            #### 读取 xml 文件
            xml_file = open(detail_xml_path)
            tree = ET.parse(xml_file)
            root = tree.getroot()
            content_dealed = []
            for obj in root.iter('object'):
                xml_writer.write('\t<object>\n')
                name = obj.find('name').text
                # print('--name:',name)
                xml_writer.write('\t\t<name>%s</name>\n'%(name))
                xmlbox = obj.find('bndbox')
                ###### 原图上的点
                x0 = int(xmlbox.find('xmin').text)
                y0 = int(xmlbox.find('ymin').text)
                x1 = int(xmlbox.find('xmax').text)
                y1 = int(xmlbox.find('ymax').text)
                if ramdom_add_padding_flag:
                    x0 +=left_padding
                    y0 +=top_padding
                    x1 +=left_padding
                    y1 +=top_padding

                    x0_, y0_, x1_, y1_ = x0, y0, x1, y1

                ###### 对应透射变换后的点
                if ramdom_perspectiveTransform_flag:
                    point_1 = np.array([[x0, y0]], dtype='float32')
                    point_1 = np.array([point_1])
                    dst = cv2.perspectiveTransform(point_1, M)
                    x0_ = int(dst[0][0][0])
                    y0_ = int(dst[0][0][1])
                    point_2 = np.array([[x1,y1]],dtype='float32')
                    point_2 = np.array([point_2])
                    dst = cv2.perspectiveTransform(point_2,M)
                    x1_= int(dst[0][0][0])
                    y1_ = int(dst[0][0][1])
                else:
                    x0_, y0_, x1_, y1_ = x0, y0, x1, y1
                ####### 修改原数据
                xml_writer.write('\t\t<bndbox>\n')
                xml_writer.write('\t\t\t<xmin>%s</xmin>\n' % (str(x0_)))
                xml_writer.write('\t\t\t<ymin>%s</ymin>\n' % (str(y0_)))
                xml_writer.write('\t\t\t<xmax>%s</xmax>\n' % (str(x1_)))
                xml_writer.write('\t\t\t<ymax>%s</ymax>\n' % (str(y1_)))
                xml_writer.write('\t\t</bndbox>\n')
                xml_writer.write('\t</object>\n')
            xml_writer.write('</annotation>')


