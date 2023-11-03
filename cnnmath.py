import numpy as np
import skimage.measure as measure


def convolve(array, kernel, bias):
    num_examples = array.shape[0]

    kd = kernel.shape[0]
    conv = np.zeros(array.shape)
    cd = conv.shape[1]

    for i in range(num_examples):
        padded = np.pad(array[i], 1)
        for j in range(cd):
            for k in range(cd):
                extract = padded[j:kd+j, k:kd+k]
                value = np.sum(extract * kernel) + bias
                conv[i][j][k] = value

    return conv


def cnn_scores(inp, WF1, bF1, WF2, bF2, WOut, bOut):
    f1 = convolve(inp, WF1, bF1)
    relu1 = np.maximum(0, f1)
    p1 = measure.block_reduce(relu1, (2, 2), np.max)

    f2 = convolve(relu1, WF2, bF2)
    relu2 = np.maximum(0, f2)
    p2 = measure.block_reduce(relu2, (2, 2), np.max)

    conv_out = np.append(p1, p2)
    scores = np.dot(conv_out, WOut) + bOut
    return scores
