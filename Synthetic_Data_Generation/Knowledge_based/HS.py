import numpy as np
import random   

def hs_transform(dataset, X, y):
    """
    consider the order of original and augmented samples
    Parameters
    ----------
    X: input original EEG signals
    y: the corresponding labels

    Returns
    X_la: left aug samples;
    X_ra: right aug samples;
    y_la, y_ra: the corresponding labels
    -------
    """
    seed = 42
    if 'BNCI2014001' in dataset:
        left_mat = [1, 2, 6, 7, 8, 13, 14, 18]
        middle_mat = [0, 3, 9, 15, 19, 21]
        right_mat = [5, 4, 12, 11, 10, 17, 16, 20]
    elif dataset == 'MI1-7':
        left_mat = [0, 2, 3, 4, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27, 33, 34, 35, 36, 41, 42, 43, 48, 49,
                    50, 55, 57]
        middle_mat = [5, 12, 28, 44, 51]
        right_mat = [1, 8, 7, 6, 15, 14, 13, 23, 22, 21, 20, 32, 31, 30, 29, 40, 39, 38, 37, 47, 46, 45, 54, 53,
                     52, 56, 58]
    elif dataset == 'BNCI2014004':
        left_mat = [0]
        middle_mat = [1]
        right_mat = [2]
    elif dataset == 'BNCI2014002':
        left_mat = [0, 3, 4, 5, 6, 12]
        right_mat = [2, 11, 10, 9, 8, 14]
        middle_mat = [1, 7, 13]
    elif dataset == 'BNCI2015001':
        left_mat = [0, 3, 4, 5, 10]
        middle_mat = [1, 6, 11]
        right_mat = [2, 9, 8, 7, 12]
    elif 'Zhou2016' in dataset:
        left_mat = [0, 2, 5, 8, 11]
        middle_mat = [3, 6, 9, 12]
        right_mat = [1, 4, 7, 10, 13]
    elif 'Weibo2014' in dataset:
        left_mat = [0, 3, 5, 6, 7, 8, 14, 15, 16, 17, 23, 24, 25, 26, 32, 33, 34, 35, 41, 42, 43, 44, 50, 51, 52, 57]
        right_mat = [2, 4, 10, 11, 12, 13, 19, 20, 21, 22, 28, 29, 30, 31, 37, 38, 39, 40, 46, 47, 48, 49, 54, 55, 56, 59]
        middle_mat = [1, 9, 18, 27, 36, 45, 53, 58]
    random.seed(seed)
    np.random.seed(seed)
    num_samples, num_channels, num_timesamples = X.shape
    llist = [i for i in range(len(y)) if y[i] == 0]
    rlist = [i for i in range(len(y)) if y[i] == 1]
    Xl = X[llist, :, :]
    Xr = X[rlist, :, :]
    Xl_left = Xl[:, left_mat, :]
    Xl_right = Xl[:, right_mat, :]
    Xl_middle = Xl[:, middle_mat, :]
    Xr_left = Xr[:, left_mat, :]
    Xr_right = Xr[:, right_mat, :]
    Xr_middle = Xr[:, middle_mat, :]
    # print(Xl_left.shape, Xl_right.shape, Xl_middle.shape, Xr_left.shape, Xr_right.shape, Xr_middle.shape)
    # input('')
    llen = list(range(0, len(llist)))
    rlen = list(range(0, len(rlist)))
    transformedL2L = np.zeros((len(llist), num_channels, num_timesamples))
    transformedR2R = np.zeros((len(rlist), num_channels, num_timesamples))
    clist = left_mat + middle_mat + right_mat
    real_list = [clist.index(h) for h in range(0, num_channels)]
    for i in range(len(llist)):
        kl = random.choice([ele for ele in llen if ele != i])
        L2L = np.concatenate((Xl_left[kl, :, :], Xl_middle[i, :, :], Xl_right[i, :, :]), axis=0)  # 右拼0类左-->0
        L2L = np.take(L2L, real_list, axis=-2)  # channel 维度重排序 0
        transformedL2L[i, :, :] = L2L
    for i in range(len(rlist)):
        kr = random.choice([ele for ele in rlen if ele != i])
        R2R = np.concatenate((Xr_left[i, :, :], Xr_middle[i, :, :], Xr_right[kr, :, :]), axis=0)  # 左拼1类右-->1
        R2R = np.take(R2R, real_list, axis=-2)  # channel 维度重排序 1
        transformedR2R[i, :, :] = R2R
    aug_X = np.concatenate((transformedL2L, transformedR2R), axis=0)
    y_la = np.zeros((transformedL2L.shape[0]))
    y_ra = np.ones((transformedR2R.shape[0]))
    aug_y = np.concatenate((y_la, y_ra), axis=0)
    return aug_X, aug_y