import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sys import argv

# ---------- Load EXR images ----------
ref = cv2.imread(argv[1], cv2.IMREAD_UNCHANGED).astype(np.float32)
test = cv2.imread(argv[2], cv2.IMREAD_UNCHANGED).astype(np.float32)

ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)
test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)

assert ref.shape == test.shape, "Images must have same resolution"

# ---------- ACES Tonemap ----------
def tonemap_aces(img):
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    return np.clip((img*(a*img+b))/(img*(c*img+d)+e), 0, 1)

# ---------- Display transform ----------
def display_transform(img, exposure):
    img = img * (2 ** exposure)          # exposure
    img = tonemap_aces(img)              # tonemap
    img = np.power(img, 1/2.2)           # gamma correction
    return np.clip(img, 0, 1)

# Automatically choose exposure from reference
exposure = -np.log2(np.mean(ref) + 1e-6) + float(argv[3] if len(argv) > 3 else 0)

ref_disp = display_transform(ref, exposure)
test_disp = display_transform(test, exposure)

# ---------- Plot ----------
fig, ax = plt.subplots(1,2, figsize=(15,5), constrained_layout=True)

ax[0].imshow(ref_disp)
ax[0].set_title("Real = float")
ax[0].axis("off")

ax[1].imshow(test_disp)
ax[1].set_title("Real = double")
ax[1].axis("off")

plt.show()