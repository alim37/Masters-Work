import cv2, numpy as np, matplotlib.pyplot as plt, os

def he(img_u8):
    L = 256
    n = np.bincount(img_u8.ravel(), minlength=L)
    cdf = n.cumsum()
    nz = cdf[cdf > 0]
    if nz.size == 0:
        return img_u8.copy()
    cdf0 = nz[0]
    s = np.rint((cdf - cdf0) / (img_u8.size - cdf0) * (L - 1)).clip(0, 255).astype(np.uint8)
    return s[img_u8]

def ahe(img_u8, tile_h=128, tile_w=128, ksize=51):
    H, W = img_u8.shape
    Hp = int(np.ceil(H / tile_h) * tile_h)
    Wp = int(np.ceil(W / tile_w) * tile_w)

    padded = cv2.copyMakeBorder(
        img_u8, 0, Hp-H, 0, Wp-W,
        borderType=cv2.BORDER_REFLECT
    )

    out = np.zeros_like(padded, dtype=np.uint8)

    for y in range(0, Hp, tile_h):
        for x in range(0, Wp, tile_w):
            out[y:y+tile_h, x:x+tile_w] = he(padded[y:y+tile_h, x:x+tile_w])

    out = out[:H, :W]

    if ksize % 2 == 0:
        ksize += 1

    out = cv2.GaussianBlur(out, (ksize, ksize), 0)

    return out


paths = [
    "/Users/malikali/Desktop/ECE552/Assignment 1/gs_extremedark.png",
    "/Users/malikali/Desktop/ECE552/Assignment 1/gs_mediumdark.png",
    "/Users/malikali/Desktop/ECE552/Assignment 1/gs_extremelight.png",
]

paths2 = [
    "/Users/malikali/Desktop/ECE552/Assignment 1/p2_extremedark.png",
    "/Users/malikali/Desktop/ECE552/Assignment 1/p2_mediumdark.png",
    "/Users/malikali/Desktop/ECE552/Assignment 1/p2_extremelight.png",
]

#part 1
for p in paths:
    assert os.path.exists(p), p
    f = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    assert f is not None, p
    L = 256
    n = np.bincount(f.ravel(), minlength=L)
    cdf = n.cumsum()
    cdf0 = cdf[cdf > 0][0]
    s = np.rint((cdf - cdf0) / (f.size - cdf0) * (L - 1)).clip(0, 255).astype(np.uint8)
    g = s[f]
    ng = np.bincount(g.ravel(), minlength=L)
    plt.figure(figsize=(10,4))
    plt.subplot(2,2,1); plt.imshow(f, cmap="gray"); plt.axis("off"); plt.title("Original")
    plt.subplot(2,2,2); plt.imshow(g, cmap="gray"); plt.axis("off"); plt.title("Equalized")
    plt.subplot(2,2,3); plt.plot(n);  plt.xlim(0,255); plt.title("Hist f(x,y)")
    plt.subplot(2,2,4); plt.plot(ng); plt.xlim(0,255); plt.title("Hist g(x,y)")
    plt.tight_layout(); plt.show()

#part 2
for p in paths2:
    assert os.path.exists(p), p
    bgr = cv2.imread(p); assert bgr is not None, p
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    R = rgb[:, :, 0]
    G = rgb[:, :, 1]
    B = rgb[:, :, 2]
    I = (R + G + B) / 3.0
    Iu = np.rint(I).clip(0, 255).astype(np.uint8)
    L = 256
    nI = np.bincount(Iu.ravel(), minlength=L)
    cdf = nI.cumsum()
    cdf0 = cdf[cdf > 0][0]
    mapI = np.rint((cdf - cdf0) / (Iu.size - cdf0) * (L - 1)).clip(0, 255).astype(np.uint8)
    Ip = mapI[Iu].astype(np.float32)
    eps = 1e-6
    ratio = Ip / (I + eps)
    Rp = (ratio * R).clip(0, 255)
    Gp = (ratio * G).clip(0, 255)
    Bp = (ratio * B).clip(0, 255)
    out = np.stack([Rp, Gp, Bp], axis=2).astype(np.uint8)
    rgb_u8 = rgb.clip(0, 255).astype(np.uint8)
    hr = np.bincount(rgb_u8[:, :, 0].ravel(), minlength=256)
    hg = np.bincount(rgb_u8[:, :, 1].ravel(), minlength=256)
    hb = np.bincount(rgb_u8[:, :, 2].ravel(), minlength=256)
    hr2 = np.bincount(out[:, :, 0].ravel(), minlength=256)
    hg2 = np.bincount(out[:, :, 1].ravel(), minlength=256)
    hb2 = np.bincount(out[:, :, 2].ravel(), minlength=256)
    plt.figure(figsize=(11,6))
    plt.subplot(2,2,1); plt.imshow(rgb_u8); plt.axis("off"); plt.title("Original")
    plt.subplot(2,2,2); plt.imshow(out);    plt.axis("off"); plt.title("Equalized (RGB via I scaling)")
    plt.subplot(2,2,3); plt.plot(hr); plt.plot(hg); plt.plot(hb); plt.xlim(0,255); plt.title("Original RGB hist")
    plt.subplot(2,2,4); plt.plot(hr2); plt.plot(hg2); plt.plot(hb2); plt.xlim(0,255); plt.title("Equalized RGB hist")
    plt.tight_layout(); plt.show()

#part 3
for p in paths2:
    assert os.path.exists(p), p
    rgb = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB).astype(np.float32)
    R, G, B = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    eps = 1e-6
    I = (R+G+B)/3.0
    S = 1 - 3*np.minimum(np.minimum(R,G),B)/(R+G+B+eps)
    num = 0.5*((R-G)+(R-B))
    den = np.sqrt((R-G)**2 + (R-B)*(G-B)) + eps
    theta = np.arccos(np.clip(num/den, -1, 1))
    H = np.where(B <= G, theta, 2*np.pi - theta)
    Iu = np.rint(I).clip(0,255).astype(np.uint8)
    L = 256
    nI = np.bincount(Iu.ravel(), minlength=L)
    cdfI = nI.cumsum()
    cdf0 = cdfI[cdfI > 0][0]
    mapI = np.rint((cdfI - cdf0)/(Iu.size - cdf0)*(L-1)).clip(0,255).astype(np.uint8)
    Ip = mapI[Iu].astype(np.float32)
    Hh = H.copy()
    Ss = np.clip(S, 0, 1)
    Ii = np.clip(Ip, 0, 255)
    R2 = np.zeros_like(Ii)
    G2 = np.zeros_like(Ii)
    B2 = np.zeros_like(Ii)
    m0 = (Hh < 2*np.pi/3)
    m1 = (Hh >= 2*np.pi/3) & (Hh < 4*np.pi/3)
    m2 = (Hh >= 4*np.pi/3)
    h0 = Hh
    B2[m0] = Ii[m0]*(1-Ss[m0])
    R2[m0] = Ii[m0]*(1 + Ss[m0]*np.cos(h0[m0])/(np.cos(np.pi/3 - h0[m0]) + eps))
    G2[m0] = 3*Ii[m0] - (R2[m0] + B2[m0])
    h1 = Hh - 2*np.pi/3
    R2[m1] = Ii[m1]*(1-Ss[m1])
    G2[m1] = Ii[m1]*(1 + Ss[m1]*np.cos(h1[m1])/(np.cos(np.pi/3 - h1[m1]) + eps))
    B2[m1] = 3*Ii[m1] - (R2[m1] + G2[m1])
    h2 = Hh - 4*np.pi/3
    G2[m2] = Ii[m2]*(1-Ss[m2])
    B2[m2] = Ii[m2]*(1 + Ss[m2]*np.cos(h2[m2])/(np.cos(np.pi/3 - h2[m2]) + eps))
    R2[m2] = 3*Ii[m2] - (G2[m2] + B2[m2])
    out = np.stack([R2,G2,B2], axis=2).clip(0,255).astype(np.uint8)
    rgb_u8 = rgb.clip(0,255).astype(np.uint8)
    hr = np.bincount(rgb_u8[:,:,0].ravel(), minlength=256)
    hg = np.bincount(rgb_u8[:,:,1].ravel(), minlength=256)
    hb = np.bincount(rgb_u8[:,:,2].ravel(), minlength=256)
    hr2 = np.bincount(out[:,:,0].ravel(), minlength=256)
    hg2 = np.bincount(out[:,:,1].ravel(), minlength=256)
    hb2 = np.bincount(out[:,:,2].ravel(), minlength=256)
    plt.figure(figsize=(11,6))
    plt.subplot(2,2,1); plt.imshow(rgb_u8); plt.axis("off"); plt.title("Original")
    plt.subplot(2,2,2); plt.imshow(out); plt.axis("off"); plt.title("HSI: Equalize I")
    plt.subplot(2,2,3); plt.plot(hr); plt.plot(hg); plt.plot(hb); plt.xlim(0,255); plt.title("Original RGB hist")
    plt.subplot(2,2,4); plt.plot(hr2); plt.plot(hg2); plt.plot(hb2); plt.xlim(0,255); plt.title("Equalized RGB hist")
    plt.tight_layout(); plt.show()

#part 4
for p in paths2:
    assert os.path.exists(p), p
    rgb = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    H = hsv[:,:,0]
    S = hsv[:,:,1]
    V = hsv[:,:,2]
    L = 256
    nV = np.bincount(V.ravel(), minlength=L)
    cdfV = nV.cumsum()
    cdf0 = cdfV[cdfV > 0][0]
    mapV = np.rint((cdfV - cdf0)/(V.size - cdf0)*(L-1)).clip(0,255).astype(np.uint8)
    Vp = mapV[V]
    out = cv2.cvtColor(np.dstack([H, S, Vp]).astype(np.uint8), cv2.COLOR_HSV2RGB)
    hr = np.bincount(rgb[:,:,0].ravel(), minlength=256)
    hg = np.bincount(rgb[:,:,1].ravel(), minlength=256)
    hb = np.bincount(rgb[:,:,2].ravel(), minlength=256)
    hr2 = np.bincount(out[:,:,0].ravel(), minlength=256)
    hg2 = np.bincount(out[:,:,1].ravel(), minlength=256)
    hb2 = np.bincount(out[:,:,2].ravel(), minlength=256)
    plt.figure(figsize=(11,6))
    plt.subplot(2,2,1); plt.imshow(rgb); plt.axis("off"); plt.title("Original")
    plt.subplot(2,2,2); plt.imshow(out); plt.axis("off"); plt.title("HSV: Equalize V")
    plt.subplot(2,2,3); plt.plot(hr); plt.plot(hg); plt.plot(hb); plt.xlim(0,255); plt.title("Original RGB hist")
    plt.subplot(2,2,4); plt.plot(hr2); plt.plot(hg2); plt.plot(hb2); plt.xlim(0,255); plt.title("Equalized RGB hist")
    plt.tight_layout(); plt.show()

#part 1 ahe
for p in paths:
    assert os.path.exists(p), p
    f = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    assert f is not None, p
    g = ahe(f, tile_h=128, tile_w=128, ksize=5)
    n = np.bincount(f.ravel(), minlength=256)
    ng = np.bincount(g.ravel(), minlength=256)
    plt.figure(figsize=(10,4))
    plt.subplot(2,2,1); plt.imshow(f, cmap="gray"); plt.axis("off"); plt.title("Original")
    plt.subplot(2,2,2); plt.imshow(g, cmap="gray"); plt.axis("off"); plt.title("AHE Equalized")
    plt.subplot(2,2,3); plt.plot(n);  plt.xlim(0,255); plt.title("Hist f(x,y)")
    plt.subplot(2,2,4); plt.plot(ng); plt.xlim(0,255); plt.title("Hist g(x,y)")
    plt.tight_layout(); plt.show()

#part 2 ahe
for p in paths2:
    assert os.path.exists(p), p
    bgr = cv2.imread(p); assert bgr is not None, p
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    R, G, B = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    I = (R+G+B)/3.0
    Iu = np.rint(I).clip(0,255).astype(np.uint8)
    Ip_u8 = ahe(Iu, tile_h=128, tile_w=128, ksize=5)
    Ip = Ip_u8.astype(np.float32)
    eps = 1e-6
    ratio = Ip / (I + eps)
    out = np.stack([(ratio * R).clip(0,255), (ratio * G).clip(0,255), (ratio * B).clip(0,255)], axis=2).astype(np.uint8)
    rgb_u8 = rgb.clip(0,255).astype(np.uint8)
    plt.figure(figsize=(11,6))
    plt.subplot(2,2,1); plt.imshow(rgb_u8); plt.axis("off"); plt.title("Original")
    plt.subplot(2,2,2); plt.imshow(out); plt.axis("off"); plt.title("AHE (RGB via I scaling)")
    plt.subplot(2,2,3)
    plt.plot(np.bincount(rgb_u8[:,:,0].ravel(), minlength=256))
    plt.plot(np.bincount(rgb_u8[:,:,1].ravel(), minlength=256))
    plt.plot(np.bincount(rgb_u8[:,:,2].ravel(), minlength=256))
    plt.xlim(0,255); plt.title("Original RGB hist")
    plt.subplot(2,2,4)
    plt.plot(np.bincount(out[:,:,0].ravel(), minlength=256))
    plt.plot(np.bincount(out[:,:,1].ravel(), minlength=256))
    plt.plot(np.bincount(out[:,:,2].ravel(), minlength=256))
    plt.xlim(0,255); plt.title("Equalized RGB hist")
    plt.tight_layout(); plt.show()

#part 3 ahe
for p in paths2:
    assert os.path.exists(p), p
    rgb = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB).astype(np.float32)
    R,G,B = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    eps = 1e-6
    I = (R+G+B)/3.0
    S = 1 - 3*np.minimum(np.minimum(R,G),B)/(R+G+B+eps)
    num = 0.5*((R-G)+(R-B))
    den = np.sqrt((R-G)**2 + (R-B)*(G-B)) + eps
    th = np.arccos(np.clip(num/den, -1, 1))
    H = np.where(B<=G, th, 2*np.pi - th)
    Iu = np.rint(I).clip(0,255).astype(np.uint8)
    Ii = ahe(Iu, tile_h=128, tile_w=128, ksize=5).astype(np.float32)
    h = H
    s = np.clip(S,0,1)
    R2 = np.zeros_like(Ii)
    G2 = np.zeros_like(Ii)
    B2 = np.zeros_like(Ii)
    m0 = (h < 2*np.pi/3)
    m1 = (h >= 2*np.pi/3) & (h < 4*np.pi/3)
    m2 = (h >= 4*np.pi/3)
    h0 = h
    B2[m0] = Ii[m0]*(1-s[m0])
    R2[m0] = Ii[m0]*(1 + s[m0]*np.cos(h0[m0])/(np.cos(np.pi/3 - h0[m0]) + eps))
    G2[m0] = 3*Ii[m0] - (R2[m0] + B2[m0])
    h1 = h - 2*np.pi/3
    R2[m1] = Ii[m1]*(1-s[m1])
    G2[m1] = Ii[m1]*(1 + s[m1]*np.cos(h1[m1])/(np.cos(np.pi/3 - h1[m1]) + eps))
    B2[m1] = 3*Ii[m1] - (R2[m1] + G2[m1])
    h2 = h - 4*np.pi/3
    G2[m2] = Ii[m2]*(1-s[m2])
    B2[m2] = Ii[m2]*(1 + s[m2]*np.cos(h2[m2])/(np.cos(np.pi/3 - h2[m2]) + eps))
    R2[m2] = 3*Ii[m2] - (G2[m2] + B2[m2])
    out = np.stack([R2,G2,B2], axis=2).clip(0,255).astype(np.uint8)
    rgb_u8 = rgb.clip(0,255).astype(np.uint8)
    plt.figure(figsize=(11,6))
    plt.subplot(2,2,1); plt.imshow(rgb_u8); plt.axis("off"); plt.title("Original")
    plt.subplot(2,2,2); plt.imshow(out); plt.axis("off"); plt.title("AHE (HSI: equalize I)")
    plt.subplot(2,2,3)
    plt.plot(np.bincount(rgb_u8[:,:,0].ravel(), minlength=256))
    plt.plot(np.bincount(rgb_u8[:,:,1].ravel(), minlength=256))
    plt.plot(np.bincount(rgb_u8[:,:,2].ravel(), minlength=256))
    plt.xlim(0,255); plt.title("Original RGB hist")
    plt.subplot(2,2,4)
    plt.plot(np.bincount(out[:,:,0].ravel(), minlength=256))
    plt.plot(np.bincount(out[:,:,1].ravel(), minlength=256))
    plt.plot(np.bincount(out[:,:,2].ravel(), minlength=256))
    plt.xlim(0,255); plt.title("Equalized RGB hist")
    plt.tight_layout(); plt.show()

#part 4 ahe
for p in paths2:
    assert os.path.exists(p), p
    rgb = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    H = hsv[:,:,0]
    S = hsv[:,:,1]
    V = hsv[:,:,2]
    Vp = ahe(V, tile_h=128, tile_w=128, ksize=5)
    out = cv2.cvtColor(np.dstack([H, S, Vp]).astype(np.uint8), cv2.COLOR_HSV2RGB)
    plt.figure(figsize=(11,6))
    plt.subplot(2,2,1); plt.imshow(rgb); plt.axis("off"); plt.title("Original")
    plt.subplot(2,2,2); plt.imshow(out); plt.axis("off"); plt.title("AHE (HSV: equalize V)")
    plt.subplot(2,2,3)
    plt.plot(np.bincount(rgb[:,:,0].ravel(), minlength=256))
    plt.plot(np.bincount(rgb[:,:,1].ravel(), minlength=256))
    plt.plot(np.bincount(rgb[:,:,2].ravel(), minlength=256))
    plt.xlim(0,255); plt.title("Original RGB hist")
    plt.subplot(2,2,4)
    plt.plot(np.bincount(out[:,:,0].ravel(), minlength=256))
    plt.plot(np.bincount(out[:,:,1].ravel(), minlength=256))
    plt.plot(np.bincount(out[:,:,2].ravel(), minlength=256))
    plt.xlim(0,255); plt.title("Equalized RGB hist")
    plt.tight_layout(); plt.show()
