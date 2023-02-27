import numpy as np 
import cv2
from sklearn.cluster import DBSCAN


def find_hull(xy, scale=100, lim=400):
    xy_ = np.int32(xy*scale)+lim
    img = np.zeros((2*lim,2*lim)).astype(np.uint8)
    img[xy_[:,1], xy_[:,0]] = 255
    img = cv2.flip(img, 0)

    return img 