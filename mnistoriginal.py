import struct
import numpy as np
import matplotlib.pyplot as plt

with open('data/MNIST/train-labels-idx1-ubyte', 'rb') as f:
    magic = struct.unpack(">I", f.read(4))

print(f"{magic:b}")


# with open('data/MNIST/train-labels-idx1-ubyte', 'rb') as f:
#     magic, size = struct.unpack(">II", f.read(8))
#     labels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
#     labels = labels.reshape(size)  # (Optional)

# print(labels)
# print(len(labels))
#
# with open('data/MNIST/train-images-idx3-ubyte', 'rb') as f:
#     magic, size = struct.unpack(">II", f.read(8))
#     nrows, ncols = struct.unpack(">II", f.read(8))
#     data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
#     data = data.reshape((size, nrows, ncols))  # (Optional)
#
# print(len(data))
# ind = 10016
# print(labels[ind])
# plt.imshow(data[ind, :, :], cmap='gray')
# plt.show()



