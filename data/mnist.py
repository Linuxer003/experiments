import numpy as np
import struct

from PIL import Image
import os

data_file = r'./input_1/MNIST/raw/t10k-images-idx3-ubyte'
data_file_size = '7840000B'
data_buf = open(data_file, 'rb').read()
magic, numImages, numRows, numColumns = struct.unpack_from('>IIII', data_buf, 0)
datas = struct.unpack_from('>'+data_file_size, data_buf, struct.calcsize('>IIII'))
datas = np.array(datas).astype(np.uint8).reshape(numImages, 1, numRows, numColumns)

label_file = r'./input_1/MNIST/raw/t10k-labels-idx1-ubyte'
label_file_size = '10000B'
label_buf = open(label_file, 'rb').read()
magic, numLabels = struct.unpack_from('>II', label_buf, 0)
labels = struct.unpack_from('>' + label_file_size, label_buf, struct.calcsize('>II'))
labels = np.array(labels).astype(np.int64)

datas_root = r'./input_1/MNIST/test'
for i in range(10):
    file_name = datas_root+os.sep+str(i)
    if not os.path.exists(file_name):
        os.mkdir(file_name)
for ii in range(numLabels):
    img = Image.fromarray(datas[ii, 0, 0:28, 0:28])
    label = labels[ii]
    file_name = datas_root+os.sep+str(label)+os.sep+str(ii)+'.png'
    img.save(file_name)
