import numpy as np
import math

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

    print(error/len(U)) # mean error. 31.99174279888267 pixels

def main():

    U = []
    V = []
    P = []

    # u, v, x, y, estimate image point
    data_path = "./data/data_2022_10_27_02_46_16.npy"
    data = np.load(data_path)
    # print(data.shape) # [.., 4] [center_x, center_y, x, y], pixel, meter.
    U = data[:, 0].astype(int)
    V = data[:, 1].astype(int)
    I = np.ones(V.shape)
    P = np.hstack([data[:, 2:], np.ones((data.shape[0], 1))])

    # print(U)
    # print(V)
    # print(I)
    # print(P)
    

    U = np.array(U) # (95,)
    V = np.array(V)
    I = np.ones(V.shape)
    # U = U.reshape((-1, 1)) 

    T = []
    T.append(cal_transform(P, U))
    T.append(cal_transform(P, V))
    T.append(cal_transform(P, I))

    print("T: ", T)

    # # # count mean error(pixel distance error)
    count_error(P, U, V, T)

if __name__ == '__main__':
    main()


