import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scripts.bev_transform import bev_tranfom, inv_bev_transform
from scripts.apply_overlay import paste_overlay

img_path = './masterscript/data/input/test/front_edge.png'
# cv2.imwrite(img_path, cv2.resize(cv2.imread(img_path), (1640, 590)))


test_overlay = {
    'width': 4,
    'length': 30,
    'blur': 0,
    'transparency': 90,
    'alpha': 0,
    'distance': 0
}

og_img, bev_img = bev_tranfom(img_path)
# plt.figure()
# plt.imshow(og_img[...,::-1])
# plt.draw()

o_img = paste_overlay(test_overlay, bev_img)
plt.figure()
plt.imshow(o_img[...,::-1])
plt.draw()

f = inv_bev_transform(og_img, o_img)
print(f.shape)




# plt.figure()
# plt.imshow(bev_img[...,::-1])
# plt.draw()

plt.figure()
plt.imshow(f[...,::-1])
plt.draw()

plt.show()