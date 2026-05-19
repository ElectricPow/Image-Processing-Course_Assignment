import sys
import cv2
import numpy as np
import skimage
import matplotlib

# print("Python 路径：", sys.executable)
# print("OpenCV 版本：", cv2.__version__)
# print("NumPy 版本：", np.__version__)
# print("scikit-image 版本：", skimage.__version__)
# print("Matplotlib 版本：", matplotlib.__version__)
a = np.random.rand(400,600,3)
b = np.max(a, axis = 2)
c = np.max(a, axis = 2, keepdims = True)
print("a shape:", a.shape)
print("b shape:", b.shape)
print("c shape:", c.shape)