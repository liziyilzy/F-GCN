import numpy as np
from Function import *
import tensorflow as tf
import torch
from Function_numpy import *
from PIL import Image
from PIL import ImageFilter
import matplotlib.image

data_name = 'UP'
Data = load_HSI_data(data_name)
site1 = Data['zuobiao']
# scio.savemat('site1', {'site1': site1})
std = 4


class FuzzyImage(object):
    def __init__(self, img):
        self.img = img
        [self.c, self.l] = np.shape(img)
        self.Dist = np.zeros([self.c, self.c])
        for i1 in range(1, self.c + 1):
            for j1 in range(1, self.c + 1):
                self.loc = site1[i1 - 1, :] - site1[j1 - 1, :]
                self.Dist[i1 - 1, j1 - 1] = np.sum(np.square(self.loc))

        scio.savemat('Dist', {'Dist': self.Dist})
        self.order_Dist = np.sort(self.Dist)
        index1 = np.argsort(self.Dist)
        scio.savemat('order_Dist', {'order_Dist': self.order_Dist})
        scio.savemat('index1', {'index1': index1})
        self.Z2 = np.zeros([self.c, self.l])
        for i2 in range(0, self.c):
            s = site1[index1[i2, 0:5], :] - site1[i2, :]
            arg = -np.sum(s * s, axis=1) / (2 * std * std)
            h = np.e ** arg
            sumh = np.sum(h)
            if sumh != 0:
                h = h / sumh
            h1 = h.reshape(-1, 1)
            # scio.savemat('h1', {'h1': h1})
            self.Z2[i2, 0:self.l] = np.sum(img[index1[i2, 0:5], :] * h1, axis=0)

        
