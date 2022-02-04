step0: 在 VOCdevkit/VOC2007/Annotations  放入 本次使用数据的 xml  文件
       在 VOCdevkit/VOC2007/JPEGImages   放入 本次使用数据的 图片 文件
step1: 修改 model_data/detect_obj_names.txt 中的类别
step2: 执行 python VOC2007/read_pic_name.py  生成训练图片的列表
step3：执行 python voc_annotation.py 生成 图片路径，标签的列表
step4：执行 python kmean.py 生成聚类的结果
step5: 执行 python train.py 开始训练

step6: 等到训练完毕可以在 logs_mobilenet/Mobilenet  获取模型