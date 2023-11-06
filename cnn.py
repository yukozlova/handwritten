import numpy as np
import cnnmath
import skimage.measure as measure
import mnistreader

N = 1
K = 10

print_counter = 0
print_every = 1

# hyperparameters
epochs = 10
reg = 0
step_size = 0.4

batches = list(mnistreader.get_train_samples(50))

# inp1 = np.array(np.mat('1 1 1 1; 1 1 1 1; 1 1 1 1; 1 1 1 1'))
# inp = np.array([inp1, inp1])
# targets = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]

# inputs_from_first_batch = batches[0][2]
# inp = np.reshape(np.array(inputs_from_first_batch), (len(inputs_from_first_batch), 28, 28))
inp_dim = 28
print(f"inp_dim = {inp_dim}")

# weights initialization
num_filters = 2
WF1 = np.random.randn(3, 3)
# WF1 = np.array(np.mat('1.0 1 1 ; 0 0 0; 0 0 0'))
WF2 = np.random.randn(3, 3)
# WF2 = np.array(np.mat('1.0 0 0 ; 1 0 0; 1 0 0'))
bF1 = 0
bF2 = 0
pool_dim = int(inp_dim / 2)
P = num_filters * pool_dim * pool_dim
WOut = np.random.randn(P, K)
# WOut = np.ones((P, K))
bOut = np.zeros((1, K))

# train the network
for epoch in range(epochs):
    for (labels, targets, inputs) in batches:
        inp = np.reshape(np.array(inputs), (len(inputs), 28, 28))
        # print(f"inp shape = {inp.shape}")
        num_examples = inp.shape[0]
        # print(f"num_examples = {num_examples}")

        # cnn calculations
        f1 = np.zeros(inp.shape)
        for i in range(num_examples):
            f1[i] = cnnmath.convolve(inp[i], WF1, bF1)
        relu1 = np.maximum(0, f1)
        # print(f"relu1 shape = {relu1.shape}")
        # print(relu1)

        f2 = np.zeros(inp.shape)
        for i in range(num_examples):
            f2[i] = cnnmath.convolve(inp[i], WF2, bF2)
        relu2 = np.maximum(0, f2)
        # print(f"relu2 shape = {relu2.shape}")
        # print(relu2)

        conv_out = np.zeros((num_examples, num_filters, inp_dim, inp_dim))
        conv_out[:, 0, :, :] = relu1
        conv_out[:, 1, :, :] = relu2

        # print(f"conv_out shape = {conv_out.shape}")
        # print(conv_out)

        pool = np.zeros((num_examples, num_filters, pool_dim, pool_dim))
        # print(f"pool shape = {pool.shape}")
        for i in range(num_examples):
            for j in range(num_filters):
                pool[i][j] = measure.block_reduce(conv_out[i][j], (2, 2), np.max)

        # print(pool)
        pool_out = np.reshape(pool, (num_examples, P))
        # print(f"pool_out shape = {pool_out.shape}")
        # print(pool_out)

        scores = np.dot(pool_out, WOut) + bOut
        # print(f"scores shape={scores.shape}")
        # print(scores)

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
        # print(f"dscores shape = {dscores.shape}, scores shape = {scores.shape}")
        # print(dscores)

        dWOut = np.dot(pool_out.T, dscores)
        # print(f"dWOut shape = {dWOut.shape}, WOut shape = {WOut.shape}")
        dbOut = np.sum(dscores, axis=0, keepdims=True)
        # print(f"dbOut shape = {dbOut.shape}, bOut shape = {bOut.shape}")

        dpool_out = np.dot(dscores, WOut.T)
        # print(f"dpool_out shape = {dpool_out.shape}, pool_out shape = {pool_out.shape}")

        dpool = np.reshape(dpool_out, (num_examples, num_filters, pool_dim, pool_dim))
        # print(f"dpool shape = {dpool.shape}, pool shape = {pool.shape}")

        dconv_out = np.zeros((num_examples, num_filters, inp_dim, inp_dim))
        for i in range(num_examples):
            for j in range(num_filters):
                cnnmath.set_dconv_out_gradient(conv_out[i][j], dconv_out[i][j], dpool[i][j])

        # print(f"dconv_out shape = {dconv_out.shape}, conv_out shape = {conv_out.shape}")
        # print(dconv_out)

        df1 = dconv_out[:, 0, :, :]
        df1[f1 <= 0] = 0
        # print(f"df1 shape = {df1.shape}, f1 shape = {f1.shape}")

        df2 = dconv_out[:, 1, :, :]
        df2[f2 <= 0] = 0
        # print(f"df2 shape = {df2.shape}, f2 shape = {f2.shape}")

        # print(f"df1[0] shape = {df1[0].shape}")
        # print(f"df1[1] shape = {df1[1].shape}")

        dWF1 = np.zeros(WF1.shape)
        for i in range(num_examples):
            dWF1_one = cnnmath.convolve(inp[i], df1[i], 0)
            dWF1 = np.add(dWF1, dWF1_one)

        # print(f"dWF1 shape = {dWF1.shape}, WF1 shape = {WF1.shape}")
        # print(dWF1)
        dbF1 = np.sum(df1)
        # print(f"dbF1 = {dbF1:.4f}, bF1 is a number")

        dWF2 = np.zeros(WF2.shape)
        for i in range(num_examples):
            dWF2_one_example = cnnmath.convolve(inp[i], df2[i], 0)
            dWF2 += dWF2_one_example

        # print(f"dWF2 shape = {dWF2.shape}, WF2 shape = {WF2.shape}")
        # print(dWF2)
        dbF2 = np.sum(df2)
        # print(f"dbF2 = {dbF2:.4f}, bF2 is a number")

        dWF1 += reg * WF1
        dWF2 += reg * WF2
        dWOut += reg * WOut

        # update weights and biases
        WF1 += -step_size * dWF1
        bF1 += -step_size * dbF1

        WF2 += -step_size * dWF2
        bF2 += -step_size * dbF2

        WOut += -step_size * dWOut
        bOut += -step_size * dbOut
