import  matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import  datasets,linear_model
import os
import time
import scipy
import  scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D
import mat4py as mp

ROOT_DIR = "./"
N_FUM = 12894
N_MESH = 4308
V_NUM = 6449

def get_data():
    print("[**] get data ......")
    start = time.time()
    deform = np.load(open(os.path.join(ROOT_DIR, "deform.npy"), "rb"))
    measure = np.load(open(os.path.join(ROOT_DIR, "measures.npy"), "rb"))
    measure = np.transpose(measure)
    dets = np.load(open(os.path.join(ROOT_DIR, "dets.npy"), "rb"))
    print("[**] deform shape:"+ str(deform.shape))
    print("[**] measure shape:"+str(measure.shape))
    print("[**] dets shape:"+str(dets.shape))

    P = measure
    Q = deform
    M = np.zeros((N_FUM,9,9))

    aM = np.zeros((N_FUM,9,9))
    bM = np.zeros((N_FUM,9))
    for i in range(0,N_FUM):
        Qtemp = np.array(Q[:,i,:])
        print("[**] get %s facet's Q ..."%(i))
        regr = linear_model.LinearRegression()
        regr.fit(measure, Qtemp)
        a, b = regr.coef_, regr.intercept_
        aM[i,:,:] = a
        bM[i,:] = b

    print("[**] aM shape:"+ str(np.array(aM).shape))
    print("[**] bM shape:" + str(np.array(bM).shape))

    new_measure = np.array([234.93966307, 406.83976607, 635.5482939, 1059.62102311, 914.03590583,
                            996.93160771, 285.82211908, 208.19797166, 1814.9772501])
    new_measure = new_measure.reshape(-1, 1)
    print("[**] new_measure shape:" + str(new_measure.shape))

    QQ = np.zeros((N_FUM,9))
    for i in range(0, N_FUM):
        newQ = np.array(aM[i,:,:]).dot(new_measure)+np.array(bM[i,:]).reshape(-1, 1)
        print("[**] solve %s  facet's Q ......" %(i))
        QQ[i,:] = np.array(newQ).flat
    print("[**] QQ shape:" + str(QQ.shape))

    return QQ




# synthesize a body by deform-based, given deform, output vertex
def d_synthesize(deform,d2v):
    d = np.array(deform.flat).reshape(deform.size, 1)
    Atd = d2v.transpose().dot(d)
    lu = scipy.sparse.linalg.splu(d2v.transpose().dot(d2v).tocsc())
    x = lu.solve(Atd)
    x = x[:V_NUM * 3]
    # move to center
    x.shape = (V_NUM, 3)
    x_mean = np.mean(x, axis=0)
    x -= x_mean
    return x

def show_mesh(points, shapeform='mesh'):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    if shapeform == 'model':
        x -= x.mean()
        y -= y.mean()
        z -= z.mean()
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z, c='r')
    n = np.arange(1, len(points), 100)
    # plt.axis('equal')
    for i, txt in enumerate(n):
        ax.text(x[txt], y[txt], z[txt], txt)

    fig.savefig('model.png');
    plt.show()

if __name__ == "__main__":
    deform  = get_data()
    loader = np.load(os.path.join(ROOT_DIR, "d2v.npz"))
    d2v = scipy.sparse.coo_matrix((loader['data'], (loader['row'], loader['col'])),shape=loader['shape'])
    x = d_synthesize(deform,d2v)
    show_mesh(x)
    # mp.savemat("newbody.mat",{'points':x})
    sio.savemat("newbody.mat",{'points':x})