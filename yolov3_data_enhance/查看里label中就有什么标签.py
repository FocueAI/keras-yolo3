import os
import xml.etree.ElementTree as ET
import shutil
from tqdm import tqdm


obj_name = []
xml_path = r'F:\各种公司\jst2\第2批数据\all_img\traindataset\label'
for index,i in tqdm(enumerate(os.listdir(xml_path))):
    detail_xml_path = xml_path + '/'+ i
    xml_file_content = open(detail_xml_path,encoding='UTF-8')
    tree = ET.parse(xml_file_content)
    root = tree.getroot()

    for obj in root.iter('object'):
        xmlbox = obj.find('bndbox')
        boxname = obj.find('name').text.strip().replace(' ','')
        print(boxname)
        obj_name.append(boxname)
print(set(obj_name))

new_object_name = list(set(obj_name))
print(list(set(obj_name)))
print(len(list(set(obj_name))))


with open('./object_name.txt','w', encoding='utf-8') as writer:
    for i in new_object_name:
        want_write_content = i + '\n'
        writer.write(want_write_content)



