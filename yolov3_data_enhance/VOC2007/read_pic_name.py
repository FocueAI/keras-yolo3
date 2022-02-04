import os
from tqdm import tqdm

with open('train.txt', 'w') as f:
    file = os.listdir('./JPEGImages')
    temp = []
    for name in file:
        temp.append(name)
    for i in tqdm(range(len(temp))):
        f.write(temp[i])
        f.write('\n')

    f.close()