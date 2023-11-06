import numpy as np
import skimage.measure as measure


def convolve_old(array, kernel, bias):
    num_examples = array.shape[0]

    kd = kernel.shape[0]
    conv = np.zeros(array.shape)
    cd = conv.shape[1]

    for i in range(num_examples):
        padded = np.pad(array[i], 1)
        for j in range(cd):
            for k in range(cd):
                extract = padded[j:kd + j, k:kd + k]
                value = np.sum(extract * kernel) + bias
                conv[i][j][k] = value

    return conv


def convolve(array, kernel, bias, padding, conv_dim):
    kernel_dim = kernel.shape[0]

    convolution = np.zeros((conv_dim, conv_dim))
    padded = np.pad(array, padding)

    for i in range(conv_dim):
        for j in range(conv_dim):
            extract = padded[i:kernel_dim + i, j:kernel_dim + j]
            value = np.sum(extract * kernel) + bias
            convolution[i][j] = value
    return convolution


def set_dconv_out_gradient(conv_out, dconv_out, dpool):
    dim = dpool.shape[0]
    for i in range(dim):
        for j in range(dim):
            (k, m) = arg2max(conv_out, i, j)
            dconv_out[k][m] = dpool[i][j]


def arg2max(conv, i, j):
    k = 2 * i
    m = 2 * j
    max_value = conv[k, m]
    res = (k, m)
    if max_value < conv[k, m + 1]:
        max_value = conv[k, m + 1]
        res = (k, m + 1)
    if max_value < conv[k + 1, m]:
        max_value = conv[k + 1, m]
        res = (k + 1, m)
    if max_value < conv[k + 1, m + 1]:
        max_value = conv[k + 1, m + 1]
        res = (k + 1, m + 1)
    # print(f"max_value for (i,j) = ({i}, {j}) is {max_value}, res = {res}")
    return res


def cnn_scores(inp, WF1, bF1, WF2, bF2, WOut, bOut):
    f1 = convolve_old(inp, WF1, bF1)
    relu1 = np.maximum(0, f1)
    p1 = measure.block_reduce(relu1, (2, 2), np.max)

    f2 = convolve_old(relu1, WF2, bF2)
    relu2 = np.maximum(0, f2)
    p2 = measure.block_reduce(relu2, (2, 2), np.max)

    conv_out = np.append(p1, p2)
    scores = np.dot(conv_out, WOut) + bOut
    return scores
