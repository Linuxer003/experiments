import os
import shutil
import random
from flbnn.net import *

# path_back = r'/home/data/backdoor/'
# file_back = os.listdir(path_back)
#
# for x in file_back:
#     shutil.copyfile(path_back+x, r'/home/data/client_0_backdoor/2/'+x)
net = ResNet18()
for name, data in net.state_dict().items():
    print(name, '\t', data.data.dtype)