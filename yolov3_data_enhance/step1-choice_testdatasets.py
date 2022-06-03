import os, shutil
from glob import glob
import random

choice_testsets_path = r'F:\各种公司\jst2\第3批数据\fix_data\fix_data\testdataset'
os.makedirs(choice_testsets_path,exist_ok=True)

# img_path1 = r'F:\各种公司\jst2\第2批数据\all_img\all_img\*.png'
img_path2 = r'F:\各种公司\jst2\第3批数据\fix_data\fix_data\img\*.jpg'
label_path = r'F:\各种公司\jst2\第3批数据\fix_data\fix_data\label'

img_list = glob(img_path2) #+ glob(img_path2)
random.shuffle(img_list)
print(len(img_list))

testdatasets_ratio = 0.1 # 挑选的测试集 占 总样本数量的 比例

for i in img_list[:int(0.1 * len(img_list))]:
    dir_path, full_file_name = os.path.split(i)
    file_name, extend_name = os.path.splitext(full_file_name)
    # print(f'file_name:{file_name}')
    # print(f'file_extend_name:{file_extend_name}')
    src_img_path = i
    src_xml_path = os.path.join(label_path,file_name + '.xml')

    # print(f'{os.path.split(i)}')
    dst_img_path = os.path.join(choice_testsets_path,os.path.basename(src_img_path))
    dst_label_path = os.path.join(choice_testsets_path, os.path.basename(src_xml_path))
    #
    shutil.move(src_img_path, dst_img_path)
    shutil.move(src_xml_path, dst_label_path)


