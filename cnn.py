import numpy as np
import cnnmath
import skimage.measure as measure

N = 1
K = 10

print_counter = 0
print_every = 1

# hyperparameters
epochs = 1
reg = 0

# inp1 = np.array(np.mat('1 1 -1 1; -1 0 1 1; 1 0 -1 1; 0 0 -1 -1'))
inp1 = np.array(np.mat('1 1 1 1; 1 1 1 1; 1 1 1 1; 1 1 1 1'))
inp = np.array([inp1, inp1])
inp_dim = 4
print(f"inp shape = {inp.shape}")
print(inp)
targets = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]

# weights initialization
num_filters = 2
# WF1 = np.random.randn(3, 3)
WF1 = np.array(np.mat('1 1 1 ; 0 0 0; 0 0 0'))
# WF2 = np.random.randn(3, 3)
WF2 = np.array(np.mat('1 0 0 ; 1 0 0; 1 0 0'))
bF1 = 0
bF2 = 0
pool_dim = int(inp_dim / 2)
# WOut = np.random.randn(pool_dim, K)
WOut = np.ones((inp_dim * pool_dim, K))
bOut = np.zeros((1, K))

# train the network
for epoch in range(epochs):
    num_examples = inp.shape[0]

    # cnn calculations
    f1 = cnnmath.convolve(inp, WF1, bF1)
    relu1 = np.maximum(0, f1)
    print(f"relu1 shape = {relu1.shape}")

    f2 = cnnmath.convolve(inp, WF2, bF2)
    relu2 = np.maximum(0, f2)
    print(f"relu2 shape = {relu2.shape}")

    conv_out = np.zeros((num_examples, num_filters, inp_dim, inp_dim))
    conv_out[:, 0, :, :] = relu1
    conv_out[:, 1, :, :] = relu2
    print("relu1")
    print(relu1)
    print('\n')
    print("relu2")
    print(relu2)
    print('\n')
    print(f"conv_out shape = {conv_out.shape}")
    print(conv_out)

    pool = np.zeros((num_examples, num_filters, pool_dim, pool_dim))
    print(f"pool shape = {pool.shape}")
    for i in range(num_examples):
        for j in range(num_filters):
            pool[i][j] = measure.block_reduce(conv_out[i][j], (2, 2), np.max)

    print(pool)
    pool_out = np.reshape(pool, (num_examples, inp_dim * pool_dim))
    print(f"pool_out shape = {pool_out.shape}")
    print(pool_out)

    scores = np.dot(pool_out, WOut) + bOut
    print(f"scores shape={scores.shape}")
    print(scores)
    exit()

    # transform scores to probabilities
    exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # loss function
    y = np.array(targets)
    correct_probs = np.max(probs*y, axis=1, keepdims=True)
    correct_log_probs = -np.log(correct_probs)

    data_loss = np.sum(correct_log_probs) / num_examples
    reg_loss = 0.5 * reg * (np.sum(WF1*WF1) + np.sum(WF2*WF2) + np.sum(WOut*WOut))
    cost = data_loss + reg_loss

    if print_counter % print_every == 0:
        print(f"\nEpoch = {epoch}, N = {num_examples}, Cost = {cost:.4f}\n")
    print_counter += 1

    # gradients
    dscores = (probs - y) / num_examples

    dWOut = np.dot(pool_out.T, dscores)
    dbOut = np.sum(dscores, axis=0, keepdims=True)

    dpool_out = np.dot(dscores, WOut.T)
    dpool = np.reshape(dpool_out, (num_examples, int(inp.shape[1] / 2), int(inp.shape[2] / 2)))

    df2 = dpool
    df2[pool <= 0] = 0

    dWF2 = cnnmath.convolve(relu1, df2, 0)
    print(f"dWF2 shape = {dWF2.shape}")
    dbF2 = np.sum(df2)

    drelu1 = cnnmath.convolve(np.pad(df2, 1), np.rot90(WF2, 2), 0)


    print(f"drelu1 shape = {drelu1.shape}")




