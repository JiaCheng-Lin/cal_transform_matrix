## regression ref: https://matters.news/@CHWang/92854-machine-learning-linear-regression%E8%BF%B4%E6%AD%B8%E6%A8%A1%E5%9E%8B-%E5%BC%B7%E5%A4%A7%E7%9A%84sklearn-%E7%B0%A1%E5%96%AE%E7%B7%9A%E6%80%A7%E8%BF%B4%E6%AD%B8%E6%A8%A1%E5%9E%8B-%E5%A4%9A%E9%A0%85%E5%BC%8F%E8%BF%B4%E6%AD%B8%E6%A8%A1%E5%9E%8B-%E5%A4%9A%E5%85%83%E8%BF%B4%E6%AD%B8%E6%A8%A1%E5%9E%8B-%E5%AE%8C%E6%95%B4%E5%AF%A6%E4%BD%9C%E6%95%99%E5%AD%B8-bafyreidbro25dokrljhrdiwfsyc2gg345yiym2uirxu4l3nc6zbxntdaqq
import os
import random

from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # split data to train&test
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures # Polynomial
from sklearn.pipeline import make_pipeline
from mpl_toolkits.mplot3d import Axes3D

from joblib import dump, load # save and load model.


### cal 2d pts l2 norm (distance)
def cal_error(pred, gt): # [[], [], ...], [[], [], ...],
    error, max_err, min_err = 0, 0, np.inf
    for i, p in enumerate(pred):
        diff = p-gt[i]
        dis = np.linalg.norm(diff) # dis between 2 pts, l2 norm
        max_err = max(dis, max_err)
        min_err = min(dis, min_err)
        error += dis 

    mean_error = error/pred.shape[0]

    return mean_error, max_err, min_err

### general linear regression
def linear_regression(X, y):
    print("linear_regression")
    # reg = LinearRegression().fit(X, y)
    reg = LinearRegression()
    reg.fit(X, y)
    
    print(reg.coef_)
    print(reg.intercept_)

    pred = reg.predict(X)
    # gt = np.array([[-0.08]])

    print(pred)

    return 0

### another linear regression "Ridge" (avoid overfitting)
def Ridge_regression(X, y):
    print("Ridge_regression")
    n_samples, n_features = 10, 5
    rng = np.random.RandomState(0)
    y = rng.randn(n_samples)
    X = rng.randn(n_samples, n_features)
    clf = Ridge(alpha=1.0)
    clf.fit(X, y) 
    # print("Ridge_regression")
    # print(clf.coef_)
    # print(clf.intercept_)

    return 0

def poly_regression(X, y, poly_degree, vis=True):
    print("poly_regression ", end='')
    # # train a polynomial regression model, 
    # 
    regressor = make_pipeline( # make a machine learning pipeline
        PolynomialFeatures(degree=poly_degree, interaction_only=False), # # interaction_only: True -> only a*b, no a^2
        Ridge(alpha=1)) # LinearRegression() can be replaced by Ridge(alpha=1), Lasso() (avoid overfitting)

    regressor.fit(X, y) # train

    # pred_y = regressor.predict(np.array([[-0.25, 1.3]])) # test
    pred_y = regressor.predict(X) # 

    mean_err, max_err, min_err = cal_error(pred_y, y)
    print("error:", mean_err, max_err, min_err)

    if vis: # True if show the vis
        vis_data(X, pred_y, "poly_"+str(poly_degree))
    
    return regressor, mean_err, max_err, min_err

### visualize 2d pts
def visualization(pts):
    x, y = pts[:, 0].reshape(-1,1), pts[:, 1].reshape(-1,1)

    plt.scatter(x, y)
    plt.show()

### visualize 4d pts. # the whole dataset.
def vis_data(UV, XY, save_info):
    u, v = UV[:, 0].reshape(-1,1), UV[:, 1].reshape(-1,1)
    x, y = XY[:, 0].reshape(-1,1), XY[:, 1].reshape(-1,1)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    img = ax.scatter(u, v, x, c=y, cmap=plt.hot())
    fig.colorbar(img)
    plt.show()

    fig.savefig(vis_sp+data_name+"/"+save_info+'.png')

def main():
    random.seed(8787)

    # (u, v), (x, y), estimate image point
    data_path = "./data/data_2022_11_02_20_07_45.npy"
    global data_name
    data_name = data_path.split("/")[-1][:-4]
    global vis_sp # sp: save path
    vis_sp ="./data/vis/"
    
    if not os.path.exists(vis_sp+data_name):
        os.makedirs(vis_sp+data_name)
 
    data = np.load(data_path) 
    # print(data.shape) # [.., 4] [center_x, center_y, x, y], pixel, meter.
    UV = data[:, :2].astype(int) # image pt
    XY = data[:, 2:] # mmwave pt

    print("UV", UV)
    print("XY", XY)

    ### Visualization original data
    # vis_data(XY, UV, "origin")
    # visualization(UV)
    # visualization(XY)

    ### poly regression
    final_mean_err = np.inf
    for degree in np.arange(2, 6, 1): # 
        regressor, mean_err, max_err, min_err = poly_regression(XY, UV, poly_degree=degree, vis=False)
        if mean_err < final_mean_err:
            final_reg = regressor
            final_mean_err = mean_err
            final_max_err = max_err
            final_min_err = min_err
    print("degree:", final_reg.steps[0][1].degree)
    print("error:", final_mean_err, final_max_err, final_min_err)
    dump(regressor, './data/'+data_name+'.joblib')


    ### test load model
    regressor = load('./data/'+data_name+'.joblib') 
    pred_y = regressor.predict(np.array([[-0.44, 1.68]])) # test
    print(pred_y) ## gt: [[550, 472]]
    


    ### linear regression 
    # T = linear_regression(XY, UV)
    # T = Ridge_regression(XY, UV)
   
    # # U = U.reshape((-1, 1))   

    # """ General method"""
    # T = []
    # T.append(cal_transform(P, U))
    # T.append(cal_transform(P, V))
    # T.append(cal_transform(P, I))
    # print("T: ", T)
    # # # # count mean error(pixel distance error)
    # mean_error = count_error(P, U, V, T) # # 40.58726889215359
    # print("mean_error", mean_error)
    # print("==============")

    # """ RANSAC(RANdom SAmple Consensus) """
    # ### using RANSAC method to remove outliers (reduce error)
    # T, min_err = RANSAC(U, V, I, P)
    # print("T", T)
    # print("min_error:", min_err)


# import numpy as np
# import math
# import random


# def RANSAC(U, V, I, P):
#     T_best = []
#     num_pts = U.shape[0]
#     k_list = [4, 8, 12, 20, 40] # sample k idx randomly.
#     min_err = np.inf

#     for k in k_list:
#         for _ in range(1000): # iteration: 1000
#             T = []
#             rand_idx = random.sample(range(num_pts), k)

#             # # cal the T
#             T.append(cal_transform(P[rand_idx], U[rand_idx]))
#             T.append(cal_transform(P[rand_idx], V[rand_idx]))
#             T.append(cal_transform(P[rand_idx], I[rand_idx]))

#             # # cal error of ALL U, V, I, P
#             mean_error = count_error(P, U, V, T)
#             if mean_error < min_err:
#                 min_err = mean_error
#                 T_best = T

#     return T, min_err



# def main():
#     random.seed(8787)

#     U = []
#     V = []
#     P = []

#     # u, v, x, y, estimate image point
#     data_path = "./data/data_2022_10_27_02_46_16.npy"
#     data = np.load(data_path)
#     # print(data.shape) # [.., 4] [center_x, center_y, x, y], pixel, meter.
#     U = data[:, 0].astype(int)
#     V = data[:, 1].astype(int)
#     I = np.ones(V.shape)
#     P = np.hstack([data[:, 2:], np.ones((data.shape[0], 1))])

#     U = np.array(U) # (95,)
#     V = np.array(V)
#     I = np.ones(V.shape)
#     # U = U.reshape((-1, 1))   

#     """ General method"""
#     T = []
#     T.append(cal_transform(P, U))
#     T.append(cal_transform(P, V))
#     T.append(cal_transform(P, I))
#     print("T: ", T)
#     # # # count mean error(pixel distance error)
#     mean_error = count_error(P, U, V, T) # # 40.58726889215359
#     print("mean_error", mean_error)
#     print("==============")

#     """ RANSAC(RANdom SAmple Consensus) """
#     ### using RANSAC method to remove outliers (reduce error)
#     T, min_err = RANSAC(U, V, I, P)
#     print("T", T)
#     print("min_error:", min_err)


if __name__ == '__main__':
    main()


