import numpy as np
import math
import random

def cal_transform(P, U):
    P_t = np.transpose(np.array(P))
    res = P_t @ P # shape: 3*3
    res_inv = np.linalg.inv(res) # shape: 3*3
    res_1 = np.matmul(res_inv, P_t) # shape: 3*95, 3*n
    t = res_1.dot(U)# np.matmul(num_1, U) # shape: 1*3 

    return t.tolist()

def count_error(P, U, V, T):
    error = 0
    for i, p in enumerate(P):
        esti_u, esti_v, _ = np.matmul(T, p)

        error += math.sqrt((esti_u-U[i])**2 + (esti_v-V[i])**2)
    mean_error = error/len(U)

    return mean_error

def RANSAC(U, V, I, P):
    T_best = []
    num_pts = U.shape[0]
    k_list = [4, 8, 12, 20, 40] # sample k idx randomly.
    min_err = np.inf

    for k in k_list:
        for _ in range(1000): # iteration: 1000
            T = []
            rand_idx = random.sample(range(num_pts), k)

            # # cal the T
            T.append(cal_transform(P[rand_idx], U[rand_idx]))
            T.append(cal_transform(P[rand_idx], V[rand_idx]))
            T.append(cal_transform(P[rand_idx], I[rand_idx]))

            # # cal error of ALL U, V, I, P
            mean_error = count_error(P, U, V, T)
            if mean_error < min_err:
                min_err = mean_error
                T_best = T

    return T, min_err



def main():
    random.seed(8787)

    U = []
    V = []
    P = []

    # u, v, x, y, estimate image point
    data_path = "./data/data_2022_11_02_20_07_45.npy"
    data = np.load(data_path)
    # print(data.shape) # [.., 4] [center_x, center_y, x, y], pixel, meter.
    U = data[:, 0].astype(int)
    V = data[:, 1].astype(int)
    I = np.ones(V.shape)
    P = np.hstack([data[:, 2:], np.ones((data.shape[0], 1))])

    U = np.array(U) # (95,)
    V = np.array(V)
    I = np.ones(V.shape)
    # U = U.reshape((-1, 1))   

    """ General method"""
    T = []
    T.append(cal_transform(P, U))
    T.append(cal_transform(P, V))
    T.append(cal_transform(P, I))
    print("T: ", T)
    # # # count mean error(pixel distance error)
    mean_error = count_error(P, U, V, T) # # 40.58726889215359
    print("mean_error", mean_error)
    print("==============")

    """ RANSAC(RANdom SAmple Consensus) """
    ### using RANSAC method to remove outliers (reduce error)
    T, min_err = RANSAC(U, V, I, P)
    print("T", T)
    print("min_error:", min_err)


if __name__ == '__main__':
    main()


