from yolo_Mobilenet_pb import YOLO
# from yolo_Mobilenet import YOLO
# from yolo import YOLO
import sys
import os,cv2,shutil
from PIL import Image
import numpy as np

#####  经过 测试 本程序 可以使用pb模型预测



if __name__=="__main__":
    yolo=YOLO(use_pb=False)
    cap = cv2.VideoCapture(0)
    ###########################
    # yolo.convert_model_to_pb('./food_loc_val_loss_6.792.pb')
    ###########################
    while True:
        sucess, img = cap.read()
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        image = yolo.detect_image(img_path=None,img=pil_img)
        res_img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        cv2.imshow("img", res_img)
        k = cv2.waitKey(1)
        if k == 27:
            # 通过esc键退出摄像
            cv2.destroyAllWindows()
    cap.release()









