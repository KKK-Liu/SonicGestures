import numpy as np

arr = np.load('./data/test/down/2023-05-19 19-58-24.npy')
print(arr.shape)
arr = np.load('./data/test/empty/2023-05-19 19-58-24.npy')
print(arr.shape)
arr = np.load('./data/train/empty/2023-05-19 19-58-24.npy')
print(arr.shape)
# arr = np.load('./data/train/up/2023-05-19 19-27-14.npy')
# print(arr.shape)

# arr = np.load('./data/test/down/2023-05-19 19-27-14.npy')
# print(arr.shape)
# arr = np.load('./data/test/up/2023-05-19 19-27-14.npy')
# print(arr.shape)