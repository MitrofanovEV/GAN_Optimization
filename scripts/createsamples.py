import sys
import numpy as np
import torch
from torch.autograd import Variable
from arch import Generator
import cv2 as cv
from PIL import Image

model_name = sys.argv[1]
n_img = int(sys.argv[2])
folder_name = sys.argv[3]
generator = Generator(3)
generator.load_state_dict(torch.load(model_name))
n = n_img // 64
k = n_img % 64
j = 0
for i in range(n):
    test_noise = Variable(torch.randn((64, 100, 1, 1), requires_grad=False).cpu())
    test_images = Variable(generator(test_noise), requires_grad=False)
    test_images = (test_images + 1.) / 2.
    imgs = test_images.cpu().detach().numpy()
    imgs = np.array(imgs * 255., dtype=int)
    imgs = np.swapaxes(imgs, 1, 3)
    imgs = np.swapaxes(imgs, 1, 2)
    for img in imgs:
        img = Image.fromarray(img.astype(np.uint8))
        img.save(folder_name+'\\{}.jpg'.format(j))
        j += 1
if k > 0:
    test_noise = Variable(torch.randn((64, 100, 1, 1), requires_grad=False).cpu())
    test_images = Variable(generator(test_noise), requires_grad=False)
    test_images = (test_images + 1.) / 2.
    imgs = test_images.cpu().detach().numpy()
    imgs = np.array(imgs * 255., dtype=int)
    imgs = np.swapaxes(imgs, 1, 3)
    imgs = np.swapaxes(imgs, 1, 2)
    print(imgs[0].shape)
    for i in range(k):
        #cv.imwrite(folder_name+'\\{}.jpg'.format(j), imgs[i])
        img = Image.fromarray(imgs[i].astype(np.uint8))
        img.save(folder_name + '\\{}.jpg'.format(j))
        j += 1


