import mnistreader
import numpy as np

batches = list(mnistreader.get_train_samples(400))

# Hyperparameters
reg = 0
step_size = 0.4
epochs = 200

D = 784
K = 10
h = 300

W = 0.01 * np.random.randn(D, h)
W2 = 0.01 * np.random.randn(h, K)
bias = np.zeros((1, h))
bias2 = np.zeros((1, K))

counter = 0

# Train the network
for epoch in range(epochs):
    for (labels, targets, inputs) in batches:
        X = np.array(inputs)
        num_examples = X.shape[0]
        inputs = np.dot(X, W) + bias
        hidden_layer = np.maximum(0, inputs)

        scores = np.dot(hidden_layer, W2) + bias2

        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        y = np.array(targets)
        correct_probs = np.max(probs * y, axis=1, keepdims=True)

        correct_logprobs = -np.log(correct_probs)
        data_loss = np.sum(correct_logprobs) / num_examples
        reg_loss = 0.5 * reg * np.sum(W*W) + 0.5 * reg * np.sum(W2*W2)
        loss_data = data_loss + reg_loss

        if counter % 50 == 0:
            print(f"Epoch = {epoch}, N = {num_examples}, Cost = {loss_data:.4f}")
        counter += 1

        dscores = (probs - y) / num_examples

        dW2 = np.dot(hidden_layer.T, dscores)
        db2 = np.sum(dscores, axis=0, keepdims=True)

        dhidden = np.dot(dscores, W2.T)
        dhidden[hidden_layer <= 0] = 0

        dW = np.dot(X.T, dhidden)
        db = np.sum(dhidden, axis=0, keepdims=True)

        dW2 += reg * W2
        dW += reg * W

        W += -step_size * dW
        bias += -step_size * db
        W2 += -step_size * dW2
        bias2 += -step_size * db2

# Test the network
(test_labels, test_targets, test_inputs) = mnistreader.get_test_samples()
X_test = np.array(test_inputs)
y_test = np.array(test_targets)
num_examples = X_test.shape[0]

hidden_layer = np.maximum(0, np.dot(X_test, W) + bias)
scores = np.dot(hidden_layer, W2) + bias2
predicted_class = np.argmax(scores, axis=1)

y_class = np.argmax(y_test, axis=1)

correct = np.count_nonzero(predicted_class == y_class)
accuracy = np.mean(predicted_class == y_class)
print(f"Accuracy: {correct}/{num_examples} ({accuracy:%})")

counter = 1
for pre, ycl, inp, la in zip(predicted_class, y_class, X_test, test_labels):
    if pre != ycl and counter > 0:
        print(f"\nPredicted: {pre}, Actual: {ycl}")
        mnistreader.plot_number(inp)
        counter -= 1

# for i in inputs:
#     i = list(i)
#     print(type(i))
#     mnisterreader.plot_number(i)
#     print("\n\n")
