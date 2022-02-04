import xml.etree.ElementTree as ET
from os import getcwd

# sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
sets=[('2007', 'train_datas')]

# classes = ['YJK_BH']
rec_classes_path = './model_data/detect_obj_names.txt'
classes = []
with open(rec_classes_path,'r',encoding='utf-8') as reader:
    contents = reader.readlines()
    for content in contents:
        classes.append(content.strip())

print('need detect obj class:',classes)

def convert_annotation(year, image_id, list_file):
    in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id),encoding='utf-8')  # add  ,encoding='utf-8'
    tree=ET.parse(in_file)
    root = tree.getroot()
    # xml_text= in_file.read()
    # root = ET.fromstring(xml_text)
    # in_file.close()
	
	
	
	

    for obj in root.iter('object'):
        # difficult = obj.find('difficult').text
        cls = obj.find('name').text
        # if cls not in classes or int(difficult)==1:
        #     continue
        cls_id = classes.index(cls)   # 在这里类别都用 0，1，2，3，4，5 的数字表示了
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

wd = getcwd()

# for year, image_set in sets:
#     image_ids1 = open('VOCdevkit/VOC%s/ImageSets/Main/train.txt'%(year)).read().strip().split()
#     image_ids2 = open('VOCdevkit/VOC%s/ImageSets/Main/problem.txt'%(year)).read().strip().split()
#     list_file = open('%s_%s.txt'%(year, image_set), 'w')
#     for image_id in image_ids1:
#         list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg'%(wd, year, image_id))
#         convert_annotation(year, image_id, list_file)
#         list_file.write('\n')
#     for image_id in image_ids2:
#         list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.png'%(wd, year, image_id))
#         convert_annotation(year, image_id, list_file)
#         list_file.write('\n')
#     list_file.close()

for year, image_set in sets:
    image_ids1 = open('VOCdevkit/VOC%s/train.txt'%(year)).read().strip().split()

    list_file = open('%s_%s.txt'%(year, image_set), 'w')
    for image_id in image_ids1:
        list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s'%(wd, year, image_id))
        # image_name = image_id.split('.')[0]
        image_name = ''
        print('image_id:',image_id)
        dealed_file = image_id.split('.')
        if len(dealed_file) < 2:
            raise TypeError
        if len(dealed_file) == 2:
            image_name = dealed_file[0]
        elif len(dealed_file) > 2:
            for no,i in enumerate(dealed_file):
                if no == len(dealed_file)-1:
                    pass
                else:
                    image_name = image_name + i + '.'
        if image_name.endswith('.'):
            image_name = image_name[:-1]
        print('image_name:',image_name)



        convert_annotation(year, image_name, list_file)
        list_file.write('\n')

    list_file.close()
