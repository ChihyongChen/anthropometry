#!/usr/bin/python
# coding=utf-8
# defomation transfer for triangle meshes： http://people.csail.mit.edu/sumner/research/deftransfer/Sumner2004DTF.pdf
import scipy
import scipy.sparse
import scipy.sparse.linalg
import random
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from multiprocessing import Pool
import numpy as np
import mat4py as mp
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import sys
import time
import os

print_switch = 0
RESULT_DIR = './result/'
MESHES_DIR = './caesar-fitted-meshes/'
MODEL_DIR = './caesar/'
ROOT_DIR = './'

F_NUM = 12894
V_NUM = 6449
M_NUM = 9
V_BASIS_NUM = 9  # 不知道啥还
D_BASIS_NUM = 9  # 不知道啥东西


# calculate global mapping from measure to deformation PCA coeff
def get_m2d(d_coeff, t_measure):
    print(' [**] begin load_m2d... ')
    start = time.time()
    D = d_coeff.copy()
    D.shape = (D.size, 1)
    M = build_equation(t_measure, D_BASIS_NUM)
    # solve transform matrix
    MtM = M.transpose().dot(M)
    MtD = M.transpose().dot(D)
    ans = np.array(scipy.sparse.linalg.spsolve(MtM, MtD))
    ans.shape = (D_BASIS_NUM, M_NUM)
    np.save(open(os.path.join(ROOT_DIR, "m2d.npy"), "wb"), ans)
    print(' [**] finish load_m2d in %fs' % (time.time() - start))
    return ans


# cosntruct the related matrix A to change deformation into vertex
def get_d2v_matrix(d_inv_mean, facet):
    print(' [**] begin reload A&lu maxtrix')
    start = time.time()
    data = []
    rowidx = []
    colidx = []
    r = 0
    off = V_NUM * 3
    shape = (F_NUM * 9, (V_NUM + F_NUM) * 3)
    for i in range(0, F_NUM):#每个面
        coeff = construct_coeff_mat(d_inv_mean[i])
        v = [c - 1 for c in facet[i, :]]
        v1 = range(v[0] * 3, v[0] * 3 + 3)
        v2 = range(v[1] * 3, v[1] * 3 + 3)
        v3 = range(v[2] * 3, v[2] * 3 + 3)
        v4 = range(off + i * 3, off + i * 3 + 3)
        for j in range(0, 3):
            data += [c for c in coeff.flat]
            rowidx += [r, r, r, r, r + 1, r + 1, r + 1,
                       r + 1, r + 2, r + 2, r + 2, r + 2]
            colidx += [v1[j], v2[j], v3[j], v4[j], v1[j],
                       v2[j], v3[j], v4[j], v1[j], v2[j], v3[j], v4[j]]
            r += 3
    d2v = scipy.sparse.coo_matrix((data, (rowidx, colidx)), shape=shape)
    np.savez(os.path.join(ROOT_DIR, "d2v"), row=d2v.row, col=d2v.col, data=d2v.data, shape=d2v.shape)
    lu = scipy.sparse.linalg.splu(d2v.transpose().dot(d2v).tocsc())
    print(' [**] finish load A&lu in %fs.' % (time.time() - start))
    return [d2v, lu]


# construct the matrix = v_mean_inv.dot(the matrix consists of 0 -1...)
def construct_coeff_mat(mat):
    tmp = -mat.sum(0)
    return np.row_stack((tmp, mat)).transpose()


# 计算面的法向 v4 = v1 +(v2 −v1)×(v3 −v1)/ sqrt( |(v2 −v1)×(v3 −v1)|)
def compute_normals(vertex, facet):
    print(vertex.shape)
    print(facet.shape)
    normals = []
    # 存放每个顶点在不同面上的法向，为了最后相加归一化
    vertexNormalLists = [[] for i in range(0, len(vertex))]
    # 遍历每个面，求他们的法向 v4 = v1 +(v2 −v1)×(v3 −v1)/ sart( |(v2 −v1)×(v3 −v1)|)
    for face in facet:
        AB = np.array(vertex[face[0]]) - np.array(vertex[face[1]])
        AC = np.array(vertex[face[0]]) - np.array(vertex[face[2]])
        n = np.cross(AB, -AC)
        n /= np.linalg.norm(n)
        for i in range(0, 3):
            # 把当前面求得的法向放到该顶点的法向列表里
            vertexNormalLists[face[i]].append(n)
    # 遍历每个顶点的法向列表
    for idx, normalList in enumerate(vertexNormalLists):
        normalSum = np.zeros(3)
        for normal in normalList:
            normalSum += normal
        normal = normalSum / float(len(normalList))
        normal /= np.linalg.norm(normal)
        normals.append(map(float, normal.tolist()))
    np.save(open(os.path.join(ROOT_DIR, "normals.npy"), "wb"), normals)

    return normals


# 计算v41 返回v=[v21 v31 v41]
def assemble_face(v1, v2, v3):
    v21 = np.array((v2 - v1))
    v31 = np.array((v3 - v1))
    v41 = np.cross(list(v21.flat), list(v31.flat))
    v41 /= np.sqrt(np.linalg.norm(v41))
    return np.column_stack((v21, np.column_stack((v31, v41))))


# 计算平均网格（model）的v的逆矩阵,第3节
def get_inv_mean(mean_vertex, facet):
    print("[**] begin get_inv_mean ...")
    start = time.time()
    d_inv_mean = np.zeros((F_NUM, 3, 3))
    for i in range(0, F_NUM):
        v = [j - 1 for j in facet[i, :]]  # 第i个面的三个顶点索引
        # 所有点列表中取出面上的三个点值
        v1 = mean_vertex[v[0], :]  # 第v[0]行
        v2 = mean_vertex[v[1], :]
        v3 = mean_vertex[v[2], :]
        d_inv_mean[i] = assemble_face(v1, v2, v3)  # 得到 v=[v21 v31 v41]
        d_inv_mean[i] = np.linalg.inv(d_inv_mean[i])  # v = v^-1 v的逆矩阵
    print(' [**] finish get_inv_mean in %fs' % (time.time() - start))
    return d_inv_mean


def print_list(list_to_print):
    for i in list_to_print:
        print("序号：%s   值：%s" % (list_to_print.index(i) + 1, i))


def load_data():
    all_girths = mp.loadmat(RESULT_DIR + 'girths.mat')
    all_girths = all_girths['girths']
    if print_switch:
        print_list(all_girths)
    all_girths = np.array(all_girths)

    all_heights = mp.loadmat(RESULT_DIR + 'heights.mat')
    all_heights = all_heights['height']
    if print_switch:
        print_list(all_heights)
        all_heights = np.array(all_heights);


def show_mesh(points, shapeform='mesh'):
    x = points[:, 0]
    #
    y = points[:, 1]
    #
    z = points[:, 2]
    #
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


def load_mesh():
    # points = mp.loadmat(MODEL_DIR+'meanShape.mat')
    # filepath = MESHES_DIR+'CSR0007A.mat'
    filepath = MODEL_DIR + 'meanShape.mat'
    points = mp.loadmat(filepath)
    points = points['points']
    points = np.array(points)
    show_mesh(points, 'model')


def save_measure():
    girths_path = RESULT_DIR + 'girths.mat'
    heights_path = RESULT_DIR + 'heights.mat'
    girths = mp.loadmat(girths_path)
    heights = mp.loadmat(heights_path)

    girths = girths['girths']
    heights = heights['height']
    heights = np.array(heights).reshape(np.array(girths).shape[0], 1)

    measures = np.column_stack((girths, heights))
    measures = np.transpose(measures)
    print(measures)
    np.save(open(os.path.join(ROOT_DIR, "measures.npy"), "wb"), measures)


def put_all_vertexes_together():
    mesh_names_path = RESULT_DIR + 'meshNames.mat'
    mesh_names = mp.loadmat(mesh_names_path)
    mesh_names = mesh_names['meshNames']
    vertex = np.zeros((len(mesh_names), V_NUM, 3), dtype=np.double)
    for i, mesh_name in enumerate(mesh_names):
        print("序号：%s  值：%s" % (i + 1, mesh_name))
        mesh_path = MESHES_DIR + mesh_name
        points = mp.loadmat(mesh_path)
        points = points['points']
        points = np.array(points)
        vertex[i, :, :] = points

    print(vertex.shape[0])
    np.save(open(os.path.join(ROOT_DIR, "all_vertex.npy"), "wb"), vertex)


def save_mean_vertex():
    vertex = np.load(open(os.path.join(ROOT_DIR, "all_vertex.npy"), "rb"))
    mean_vertex = np.array(vertex.mean(axis=0)).reshape(V_NUM, 3)
    show_mesh(mean_vertex)
    np.save(open(os.path.join(ROOT_DIR, "mean_vertex.npy"), "wb"), mean_vertex)


def save_facet():
    facets_path = ROOT_DIR + 'fitting/facesShapeModel.mat'
    facets = mp.loadmat(facets_path)
    facets = facets['faces']
    np.save(open(os.path.join(ROOT_DIR, "facets.npy"), "wb"), facets)


def save_deform():
    print("[**] begin save deform data ...")
    start = time.time()
    dets = []  # determinant
    # vertex 包含所有4308个mesh的点
    vertex = np.load(open(os.path.join(ROOT_DIR, "all_vertex.npy"), "rb"))
    mean_vertex = np.load(open(os.path.join(ROOT_DIR, "mean_vertex.npy"), "rb"))
    facet = np.load(open(os.path.join(ROOT_DIR, "facets.npy"), "rb"))
    d_inv_mean = get_inv_mean(mean_vertex, facet)  # V^-1 V的逆矩阵
    deform = np.zeros((vertex.shape[0], F_NUM, 9))

    for i in range(0, F_NUM):
        print('\r>> loading %s deformation of facet ' % (i))
        v = [k - 1 for k in facet[i, :]]  # 取一个面，facet里头点的索引是从1开始的，而保存点的索引是从0开始的
        for j in range(0, vertex.shape[0]):  # 遍历所有mesh
            v1 = vertex[j, v[0], :]
            v2 = vertex[j, v[1], :]
            v3 = vertex[j, v[2], :]
            Q = assemble_face(v1, v2, v3).dot(d_inv_mean[i])
            dets.append(np.linalg.det(Q))  # 计算每个面的秩
            Q.shape = (9, 1)
            deform[j, i, :] = Q.flat  # 保存第j个mesh的第i面的Q
    dets = np.array(dets).reshape(F_NUM, vertex.shape[0])  # m个面，n个mesh
    np.save(open(os.path.join(ROOT_DIR, "dets.npy"), "wb"), dets)
    np.save(open(os.path.join(ROOT_DIR, "d_inv_mean.npy"), "wb"), d_inv_mean)
    np.save(open(os.path.join(ROOT_DIR, "deform.npy"), "wb"), deform)
    mean_deform = np.array(deform.mean(axis=0))
    mean_deform.shape = (F_NUM, 9)
    std_deform = np.array(deform.std(axis=0))
    std_deform.shape = (F_NUM, 9)
    np.save(open(os.path.join(ROOT_DIR, "mean_deform.npy"), "wb"), mean_deform)
    np.save(open(os.path.join(ROOT_DIR, "std_deform.npy"), "wb"), std_deform)
    print('\n[**] finish save_deformation in %fs' % (time.time() - start))


# calculating vertex-based presentation(PCA) using t-vertex
def get_v_basis():
    vertex = np.load(open(os.path.join(ROOT_DIR, "all_vertex.npy"), "rb"))
    print(" [**] begin get_v_basis of %s ...")
    start = time.time()
    body_num = vertex.shape[0]  # mesh个数
    mean_vertex = np.array(vertex.mean(axis=0)).reshape(V_NUM, 3)  # 平均的mesh
    vertex -= mean_vertex  #
    std_vertex = np.array(vertex.std(axis=0)).reshape(V_NUM, 3)
    vertex /= std_vertex
    vertex.shape = (vertex.shape[0], 3 * V_NUM)
    v = vertex.transpose()
    # principle component analysis
    v_basis, v_sigma, V = np.linalg.svd(v, full_matrices=0)
    v_basis = np.array(v_basis[:, :V_BASIS_NUM]).reshape(3 * V_NUM, V_BASIS_NUM)

    # coefficient
    v_coeff = np.dot(v_basis.transpose(), v)
    v_pca_mean = np.array(np.mean(v_coeff, axis=1))
    v_pca_mean.shape = (v_pca_mean.size, 1)
    v_pca_std = np.array(np.std(v_coeff, axis=1))
    v_pca_std.shape = (v_pca_std.size, 1)
    vertex.shape = (body_num, V_NUM, 3)
    vertex *= std_vertex
    vertex += mean_vertex
    np.save(open(os.path.join(ROOT_DIR, "v_basis.npy"), "wb"), v_basis)
    np.save(open(os.path.join(ROOT_DIR, "v_coeff.npy"), "wb"), v_coeff)
    print(' [**] finish get_v_basis in %fs' % (time.time() - start))
    return [v_basis, v_coeff, v_pca_mean, v_pca_std]


# calculating deform-based presentation(PCA)
def get_d_basis():
    deform = np.load(open(os.path.join(ROOT_DIR, "deform.npy"), "rb"))
    print(" [**] deform shape:"+str(deform.shape))
    print(" [**] begin get_d_basis ...")
    start = time.time()
    body_num = deform.shape[0]
    mean_deform = np.array(deform.mean(axis=0))
    mean_deform.shape = (F_NUM, 9)
    std_deform = np.array(deform.std(axis=0))
    std_deform.shape = (F_NUM, 9)
    deform -= mean_deform
    deform /= std_deform
    deform.shape = (deform.shape[0], 9 * F_NUM)
    d = deform.transpose()

    # principle component analysis
    d_basis, d_sigma, V = np.linalg.svd(d, full_matrices=0)
    d_basis = np.array(d_basis[:, :D_BASIS_NUM]).reshape(9 * F_NUM, D_BASIS_NUM)
    print(" [**] d_basis shape:" + str(d_basis.shape))
    d_coeff = np.dot(d_basis.transpose(), d)
    d_pca_mean = np.array(np.mean(d_coeff, axis=1))
    d_pca_mean.shape = (d_pca_mean.size, 1)
    d_pca_std = np.array(np.std(d_coeff, axis=1))
    d_pca_std.shape = (d_pca_std.size, 1)

    np.save(open(os.path.join(ROOT_DIR, "d_basis.npy"), "wb"), d_basis)
    np.save(open(os.path.join(ROOT_DIR, "d_coeff.npy"), "wb"), d_coeff)
    deform.shape = (body_num, F_NUM, 9)
    deform *= std_deform
    deform += mean_deform
    print(' [**] finish get_d_basis in %fs' % (time.time() - start))
    return [d_basis, d_coeff, d_pca_mean, d_pca_std]


# build sparse matrix
def build_equation(m_datas, basis_num):
    shape = (m_datas.shape[1] * basis_num, m_datas.shape[0] * basis_num)
    data = []
    rowid = []
    colid = []
    for i in range(0, m_datas.shape[1]):  # 1531
        for j in range(0, basis_num):  # 10
            data += [c for c in m_datas[:, i].flat]
            rowid += [basis_num * i + j for a in range(m_datas.shape[0])]
            colid += [a for a in range(j * m_datas.shape[0],
                                       (j + 1) * m_datas.shape[0])]
    return scipy.sparse.coo_matrix((data, (rowid, colid)), shape)


# local map matrix: measure->deform
def local_matrix(mask,deform, measure):
    print(' [**] begin solve local_matrix')
    start = time.time()
    L_tosave = []
    body_num = deform.shape[0]
    for i in range(0, F_NUM):
        sys.stdout.write('\r>> calc local map NO.%d'%(i))
        sys.stdout.flush()
        S = np.array(deform[:, i, :])
        S.shape = (S.size, 1)
        t_mask = np.array(mask[:, i])
        t_mask.shape = (M_NUM, 1)
        t_mask = t_mask.repeat(body_num, axis=1)
        m = np.array(measure)
        m.shape = (m.size // body_num, body_num)
        M = build_equation(m, 9)
        # solve transform matrix
        MtM = M.transpose().dot(M)
        MtS = M.transpose().dot(S)
        ans = np.array(scipy.sparse.linalg.spsolve(MtM, MtS))
        ans.shape = (9, m.size // body_num)
        L_tosave.append(list(ans))
    np.save(open(os.path.join(ROOT_DIR, "local.npy"), "wb"), L_tosave)
    print('\n [**] finish solve local_matrix in %fs' % (time.time() - start))


# calculate relationship directly
def rfe_local(dets, deform, measure, k_features=9):
    print(' [**] begin rfe_local')
    start = time.time()
    body_num = deform.shape[0]
    mean_measure = np.array(measure.mean(axis=1)).reshape(M_NUM, 1)
    std_measure = np.array(measure.std(axis=1)).reshape(M_NUM, 1)
    t_measure = measure - mean_measure
    t_measure /= std_measure
    # np.save(open(os.path.join(ROOT_DIR, "std_measure.npy"), "wb"), std_measure)
    # np.save(open(os.path.join(ROOT_DIR, "t_measure.npy"), "wb"), t_measure)
    # np.save(open(os.path.join(ROOT_DIR, "mean_measure.npy"), "wb"), mean_measure)
    #
    # return
    x = t_measure.transpose()

    pool = Pool(processes=8)
    tasks = [(i, dets[i, :], deform[:, i, :], body_num, x, measure, k_features) for i in range(F_NUM)]
    results = pool.starmap(rfe_multiprocess, tasks)
    pool.close()
    pool.join()

    rfe_mat = np.array([ele[0] for ele in results]).reshape(F_NUM, 9, k_features)
    mask = np.array([ele[1] for ele in results]).reshape(F_NUM, M_NUM).transpose()

    np.save(open(os.path.join(ROOT_DIR, "rfemat.npy"), "wb"), rfe_mat)
    np.save(open(os.path.join(ROOT_DIR, "rfemask.npy"), "wb"), mask)
    print("[**] finish rfe_mat calc in %fs" % (time.time() - start))
    return [dets, mask, rfe_mat]


def rfe_multiprocess(i, dets, deform, body_num, x, measure, k_features):
    sys.stdout.write('>> calc rfe map NO.%d\n' % (i))
    y = np.array(dets).reshape(body_num, 1)
    model = LinearRegression()
    # recurcive feature elimination
    rfe = RFE(model, k_features)
    rfe.fit(x, y.ravel())
    # mask.append(rfe.support_)
    flag = np.array(rfe.support_).reshape(M_NUM, 1)
    flag = flag.repeat(body_num, axis=1)

    # calculte linear mapping mat
    S = np.array(deform)
    S.shape = (S.size, 1)
    m = np.array(measure[flag])
    m.shape = (k_features, body_num)
    M = build_equation(m, 9)
    MtM = M.transpose().dot(M)
    MtS = M.transpose().dot(S)
    ans = np.array(scipy.sparse.linalg.spsolve(MtM, MtS))
    ans.shape = (9, k_features)
    return [ans, rfe.support_]

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

# global mapping using t_measure
def mapping_global(std_deform,mean_deform,new_measure,m2d,d_basis,d2v):
    new_measure = np.array(new_measure).reshape(M_NUM, 1)
    new_measure = m2d.dot(new_measure)
    d = np.matmul(d_basis, new_measure)
    d.shape = (F_NUM, 9)
    d *= std_deform
    d += mean_deform
    d.shape = (F_NUM * 9, 1)
    v = d_synthesize(d,d2v)
    return v


# local mapping using measure + mask
def mapping_mask(weight):
    weight = np.array(weight).reshape(M_NUM, 1)
    weight *= std_measure
    weight += mean_measure
    d = []
    for i in range(0, F_NUM):
        mask = np.array(mask[:, i]).reshape(M_NUM, 1)
        alpha = np.array(weight[mask])
        alpha.shape = (alpha.size, 1)
        s = local_mat[i].dot(alpha)
        d += [a for a in s.flat]
    d = np.array(d).reshape(F_NUM * 9, 1)
    v = d_synthesize(d)
    return v

# local mapping using measure + rfe_mat
def mapping_rfemat(weight,std_measure,mean_measure,rfemask,rfemat,d2v):
    weight = np.array(weight).reshape(M_NUM, 1)
    weight *= std_measure
    weight += mean_measure
    d = []
    for i in range(0, F_NUM):
        mask = np.array(rfemask[:, i]).reshape(M_NUM, 1)
        alpha = np.array(weight[mask])
        alpha.shape = (alpha.size, 1)
        s = rfemat[i].dot(alpha)
        d += [a for a in s.flat]
    d = np.array(d).reshape(F_NUM * 9, 1)
    v = d_synthesize(d,d2v)
    return v

if __name__ == "__main__":
    # load_data()
    # load_mesh()
    # put_all_vertexes_together()
    # save_mean_vertex()
    # save_facet()
    # save_deform()
    # save_measure()
    #  get_v_basis()
    # v_coeff = np.load(open(os.path.join(ROOT_DIR, "v_coeff.npy"), "rb"))
    # v_basis = np.load(open(os.path.join(ROOT_DIR, "v_basis.npy"), "rb"))
    # print(v_coeff.shape)
    # print(v_basis.shape)
    # get_d_basis()
    # dets = np.load(open(os.path.join(ROOT_DIR, "dets.npy"), "rb"))
    # deform = np.load(open(os.path.join(ROOT_DIR, "deform.npy"), "rb"))
    # measure = np.load(open(os.path.join(ROOT_DIR, "measures.npy"), "rb"))
    # rfe_local(dets, deform, measure)
    # vertex = np.load(open(os.path.join(ROOT_DIR,"all_vertex.npy"),"rb"))
    # facets = np.load(open(os.path.join(ROOT_DIR,"facets.npy"),"rb"))
    # compute_normals(vertex,facets)

    # d_coeff = np.load(open(os.path.join(ROOT_DIR,"d_coeff.npy"),"rb"))
    # t_measure = np.load(open(os.path.join(ROOT_DIR,"t_measure.npy"),"rb"))
    #
    # get_m2d(d_coeff, t_measure)

    # d_inv_mean = np.load(open(os.path.join(ROOT_DIR, "d_inv_mean.npy"), "rb"))
    # facet = np.load(open(os.path.join(ROOT_DIR, "facets.npy"), "rb"))
    # get_d2v_matrix(d_inv_mean, facet)

    # mask = np.ones((9,F_NUM))
    # np.save(open(os.path.join(ROOT_DIR, "mask.npy"), "wb"), mask)

    # mask = np.load(open(os.path.join("./", "mask.npy"), "rb"))
    # print(mask.shape)
    # mask = np.load(open(os.path.join("./", "mask.npy"), "rb"))
    # deform = np.load(open(os.path.join("./", "deform.npy"), "rb"))
    # measure = np.load(open(os.path.join("./", "measures.npy"), "rb"))
    #
    # print("mask shape:"+ str(mask.shape))
    # print("deform shape:"+ str(deform.shape))
    # print("measure shape:"+str(measure.shape))
    # local_matrix(mask, deform, measure)

    # dets = np.load(open(os.path.join("./", "dets.npy"), "rb"))
    # print(dets.shape)
    # measure = np.load(open(os.path.join("./", "measures.npy"), "rb"))
    # print(measure.shape)
    # print(measure[:,456])
    # print("m2d shape:"+ str(measure.shape))

    #M = np.load(open(os.path.join(ROOT_DIR,"local.npy"),"rb"))
    new_measure = np.array([ 234.93966307,406.83976607 , 635.5482939 , 1059.62102311  ,914.03590583,
  996.93160771 , 285.82211908 , 208.19797166, 1814.9772501 ])
    # new_measure = new_measure.reshape(-1,1)


    m2d = np.load(open(os.path.join(ROOT_DIR,"m2d.npy"),"rb"))
    d_basis = np.load(open(os.path.join(ROOT_DIR, "d_basis.npy"), "rb"))
    std_deform = np.load(open(os.path.join(ROOT_DIR, "std_deform.npy"), "rb"))
    mean_deform = np.load(open(os.path.join(ROOT_DIR, "mean_deform.npy"), "rb"))
    loader = np.load(os.path.join(ROOT_DIR, "d2v.npz"))
    d2v = scipy.sparse.coo_matrix((loader['data'], (loader['row'], loader['col'])),shape=loader['shape'])
    # std_measure = np.load(open(os.path.join(ROOT_DIR,"std_measure.npy"),"rb"))
    # mean_measure = np.load(open(os.path.join(ROOT_DIR,"mean_measure.npy"),"rb"))
    # rfemask = np.load(open(os.path.join(ROOT_DIR,"rfemask.npy"),"rb"))
    # rfemat = np.load(open(os.path.join(ROOT_DIR,"rfemat.npy"),"rb"))

# print(std_measure.shape)
# print(std_measure)
#
# print(mean_measure.shape)
# print(mean_measure)


    v = mapping_global(std_deform,mean_deform,new_measure,m2d,d_basis,d2v)
    # v = mapping_rfemat(new_measure, std_measure, mean_measure, rfemask, rfemat,d2v)
    show_mesh(v)


