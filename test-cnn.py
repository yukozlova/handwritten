import numpy as np
import cnnmath
import skimage.measure as measure
import mnistreader

num_filters = 2
inp_dim = 28
pool_dim = int(inp_dim / 2)
P = num_filters * pool_dim * pool_dim

WF1 = np.load("weights/WF1.npy")
bF1 = np.load("weights/bF1.npy")
WF2 = np.load("weights/WF2.npy")
bF2 = np.load("weights/bF2.npy")
WOut = np.load("weights/WOut.npy")
bOut = np.load("weights/bOut.npy")

print(WF1)
print(WF2)
print(bF1)
print(bF2)
print(WOut)
print(bOut)

# Test the network
(test_labels, test_targets, test_inputs) = mnistreader.get_test_samples()
T = 2
test_labels = test_labels[:T]
test_targets = test_targets[:T]
test_inputs = test_inputs[:T]

X_test = np.reshape(np.array(test_inputs), (len(test_inputs), 28, 28))
y_test = np.array(test_targets)
print(f"y_test = {y_test}")
test_num_examples = X_test.shape[0]
print(f"test_num_examples = {test_num_examples}")

padding = int((WF1.shape[0] - 1) / 2)

f1 = np.zeros(X_test.shape)
for i in range(test_num_examples):
    f1[i] = cnnmath.convolve(X_test[i], WF1, bF1, padding, inp_dim)
relu1 = np.maximum(0, f1)

# print("\nrelu1")
# print(f1[0])
# print("\n\n")
# print(f1[1])

padding = int((WF2.shape[0] - 1) / 2)

f2 = np.zeros(X_test.shape)
for i in range(test_num_examples):
    f2[i] = cnnmath.convolve(X_test[i], WF2, bF2, padding, inp_dim)
relu2 = np.maximum(0, f2)

conv_out = np.zeros((test_num_examples, num_filters, inp_dim, inp_dim))
conv_out[:, 0, :, :] = relu1
conv_out[:, 1, :, :] = relu2

pool = np.zeros((test_num_examples, num_filters, pool_dim, pool_dim))

for i in range(test_num_examples):
    for j in range(num_filters):
        pool[i][j] = measure.block_reduce(conv_out[i][j], (2, 2), np.max)

pool_out = np.reshape(pool, (test_num_examples, P))

scores = np.dot(pool_out, WOut) + bOut

predicted_class = np.argmax(scores, axis=1)
print(f"scores = {scores}")
print(f"predicted_class = {predicted_class}")
y_class = np.argmax(y_test, axis=1)
print(f"y_class = {y_class}")

correct = np.count_nonzero(predicted_class == y_class)
accuracy = np.mean(predicted_class == y_class)
print(f"Accuracy: {correct}/{test_num_examples} ({accuracy:%})")

counter = 0
for pre, ycl, inp, la in zip(predicted_class, y_class, test_inputs, test_labels):
    if pre != ycl and counter > 0:
        print(f"\nPredicted: {pre}, Actual: {ycl}")
        mnistreader.plot_number(inp)
        counter -= 1
