# Part 1

import cv2, numpy as np, matplotlib.pyplot as plt, os

paths = [
    "/Users/malikali/Desktop/ECE552/Assignment 1/gs_extremedark.png",
    "/Users/malikali/Desktop/ECE552/Assignment 1/gs_mediumdark.png",
    "/Users/malikali/Desktop/ECE552/Assignment 1/gs_extremelight.png",
]

for p in paths:
    f = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    L = 256
    n = np.bincount(f.ravel(), minlength=L) 
    pr = n / f.size
    s = np.floor((L-1) * np.cumsum(pr)).astype(np.uint8)   
    g = s[f]                                              
    ng = np.bincount(g.ravel(), minlength=L)

    plt.figure(figsize=(10,4))
    plt.subplot(2,2,1); plt.imshow(f, cmap="gray"); plt.axis("off"); plt.title("Original")
    plt.subplot(2,2,2); plt.imshow(g, cmap="gray"); plt.axis("off"); plt.title("Equalized")
    plt.subplot(2,2,3); plt.plot(n);  plt.xlim(0,255); plt.title("Hist f(x,y)")
    plt.subplot(2,2,4); plt.plot(ng); plt.xlim(0,255); plt.title("Hist g(x,y)")
    plt.tight_layout(); plt.show()

# Part 2

paths2 = [
    "/Users/malikali/Desktop/ECE552/Assignment 1/p2_extremedark.png",
    "/Users/malikali/Desktop/ECE552/Assignment 1/p2_mediumdark.png",
    "/Users/malikali/Desktop/ECE552/Assignment 1/p2_extremelight.png",
]

for p in paths2:
    assert os.path.exists(p), p
    bgr = cv2.imread(p); assert bgr is not None, p
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    R,G,B = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    I = (R+G+B)/3.0; L=256
    Iu = np.clip(np.round(I),0,255).astype(np.uint8)

    n = np.bincount(Iu.ravel(), minlength=L); pr = n / Iu.size
    s = np.floor((L-1)*np.cumsum(pr)).astype(np.uint8)
    Ip = s[Iu].astype(np.float32)

    ratio = Ip / np.maximum(I, 1e-6)
    out = np.clip(rgb * ratio[:,:,None], 0, 255).astype(np.uint8)

    plt.figure(figsize=(11,6))
    plt.subplot(2,2,1); plt.imshow(rgb.astype(np.uint8)); plt.axis("off"); plt.title("Original")
    plt.subplot(2,2,2); plt.imshow(out); plt.axis("off"); plt.title("Equalized (RGB via I)")
    plt.subplot(2,2,3)
    plt.plot(np.bincount(rgb[:,:,0].astype(np.uint8).ravel(), minlength=256))
    plt.plot(np.bincount(rgb[:,:,1].astype(np.uint8).ravel(), minlength=256))
    plt.plot(np.bincount(rgb[:,:,2].astype(np.uint8).ravel(), minlength=256))
    plt.xlim(0,255); plt.title("Original RGB hist")
    plt.subplot(2,2,4)
    plt.plot(np.bincount(out[:,:,0].ravel(), minlength=256))
    plt.plot(np.bincount(out[:,:,1].ravel(), minlength=256))
    plt.plot(np.bincount(out[:,:,2].ravel(), minlength=256))
    plt.xlim(0,255); plt.title("Equalized RGB hist")
    plt.tight_layout(); plt.show()