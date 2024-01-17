import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

im = Image.open('test.jpg')
im_pillow = np.array(im)

im_pillow_c1 = im_pillow.copy()
im_pillow_c1[:, :, 1:] = 0
im_pillow_c2 = im_pillow.copy()
im_pillow_c2[:, :, [0, 2]] = 0
im_pillow_c3 = im_pillow.copy()
im_pillow_c3[:, :, :2] = 0

plt.subplot(2, 2, 1)
plt.title('Origin Image')
plt.imshow(im_pillow)
plt.axis('off')
plt.subplot(2, 2, 2)
plt.title('Red Channel')
plt.imshow(im_pillow_c1)
plt.axis('off')
plt.subplot(2, 2, 3)
plt.title('Green Channel')
plt.imshow(im_pillow_c2)
plt.axis('off')
plt.subplot(2, 2, 4)
plt.title('Blue Channel')
plt.imshow(im_pillow_c3)
plt.axis('off')
plt.savefig('./rgb_pillow2.png', dpi=150)

