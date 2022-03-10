import scipy.io as sio
from matplotlib import image


def svhn2conv2jpg():
    data = sio.loadmat(r'./input_1/train_32x32.mat')
    x = data['X']
    y = data['y']
    x = x.transpose(3, 0, 1, 2)
    y = y.reshape(y.shape[0])

    for i in range(73257):
        img, label = x[i], y[i]
        img_name = r'./input_1/svhn/train/'+str(label)+'/'+str(i)+'.jpg'
        image.imsave(img_name, img)

    data = sio.loadmat(r'./input_1/test_32x32.mat')
    x = data['X']
    y = data['y']
    x = x.transpose(3, 0, 1, 2)
    y = y.reshape(y.shape[0])

    for i in range(26032):
        img, label = x[i], y[i]
        img_name = r'./input_1/svhn/test/' + str(label) + '/' + str(i) + '.jpg'
        image.imsave(img_name, img)