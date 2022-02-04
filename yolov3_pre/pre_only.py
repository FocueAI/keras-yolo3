from yolo_Mobilenet_pb import YOLO
# from yolo_Mobilenet import YOLO
# from yolo import YOLO
import sys
import os,cv2,shutil
from PIL import Image

#####  经过 测试 本程序 可以使用pb模型预测

res_path = './test_datasets/out'
if os.path.exists(res_path): shutil.rmtree(res_path)
os.mkdir(res_path)

need_detect_path = r'./test_datasets'
if __name__=="__main__":
    yolo=YOLO(use_pb=False)
    ###########################
    # yolo.convert_model_to_pb('./food_loc_2021_12_20.pb')
    ###########################
    for i in os.listdir(need_detect_path):
        if i.endswith('.jpg') or i.endswith('.png'):
            detail_path = os.path.join(need_detect_path,i)
            res = yolo.detect_image(img_path=detail_path)

            save_path = os.path.join(res_path,i)
            res.save(save_path)








