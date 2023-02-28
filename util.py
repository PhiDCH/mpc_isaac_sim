import numpy as np 
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import cv2


def polar2decart(data, limit=4.0):
    rads = np.linspace(0, 2*np.pi, 1972)
    X = []
    for i, rad in enumerate(rads):
        if data[i] < limit:
            X.append([data[i]*np.cos(rad), data[i]*np.sin(rad)])
    X = np.array(X)
    return X

def fit2ellipse(hull, n_std=2.0):
    cov = np.cov(hull[:,0], hull[:,1])
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)*np.sqrt(cov[0, 0]) * n_std
    ell_radius_y = np.sqrt(1 - pearson)*np.sqrt(cov[1, 1]) * n_std

    mean_x, mean_y = np.mean(hull, axis=0)

    w,v = np.linalg.eig(cov)
    vmax = v[np.argmin(w)]
    rot = np.rad2deg(np.arctan2(vmax, [1,0])[0])
    if rot<0:
        rot += 90
    # else:
    #     rot = 90 -rot
    return [mean_x, mean_y, ell_radius_x, ell_radius_y, rot]

def findEllipse(X):
    clus = DBSCAN(eps=0.5, min_samples=5).fit(X).labels_
    ells = []
    for i in set(clus):
        xy = X[np.where(clus==i)]
        # xy = find_hull(xy)
        ells.append(fit2ellipse(xy, n_std=2.0))
    return ells

def find_hull(xy, scale=100, lim=400):
    xy_ = np.int32(xy*scale)+lim
    img = np.zeros((2*lim,2*lim)).astype(np.uint8)
    img[xy_[:,1], xy_[:,0]] = 255
    # img = cv2.flip(img, 0)

    kernel = np.ones((5,5))
    img = cv2.dilate(img, kernel, iterations=1)
    kernel = np.ones((3,3))
    img = cv2.erode(img, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 1:
        print('num of contour', len(contours))
    # img_show = cv2.drawContours(img.copy(), contours, -1, 255, 3)
    hull = cv2.convexHull(contours[0])
    # img_show = cv2.drawContours(img.copy(), [hull], -1, 255, 3)
    hull = hull.reshape((-1,2))
    hull = (hull-lim)/scale
    return hull

def testPlot(X, ells, limit=4.0):
    fig, ax = plt.subplots()
    ax.set(xlim=(-limit,limit), xticks=np.arange(-limit,limit),
            ylim=(-limit,limit), yticks=np.arange(-limit,limit))
    
    plt.scatter(X[:,0], X[:,1])
    for ell in ells:
        x,y,a,b,rot = ell
        ellipse = Ellipse((x,y), width=2*a, height=2*b, angle=rot, fill=False)
        ax.add_patch(ellipse)

    plt.show()

if __name__=='__main__':
    import time
    data = np.load('data.npy', allow_pickle=True)
    t1 = time.time()
    limit = 4.0
    X = polar2decart(data, limit)
    ells = findEllipse(X)
    print(time.time()-t1)

    testPlot(X, ells, limit)