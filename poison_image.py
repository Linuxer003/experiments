from PIL import Image
import os
import random
img_paste = Image.new('RGB', (5, 5), 'white')
t = 0
for i in range(1, 10):
    path = f'D:/data/MNIST/train/{i}/'
    files = os.listdir(path)
    files = random.sample(files, 10)
    for x in files:
        img = Image.open(path+x)
        img.paste(img_paste, (23, 23), )
        img.save(f'D:/data/MNIST/poison/{t}.png')
        t += 1
