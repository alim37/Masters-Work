import cv2, numpy as np, matplotlib.pyplot as plt, os

paths_gs = [
    r"C:\Users\Malik\OneDrive\Desktop\MASTERS\ECE 552 Visual Perception for Autonomy\Assignment 1\gs_extremedark.png",
    r"C:\Users\Malik\OneDrive\Desktop\MASTERS\ECE 552 Visual Perception for Autonomy\Assignment 1\gs_mediumdark.png",
    r"C:\Users\Malik\OneDrive\Desktop\MASTERS\ECE 552 Visual Perception for Autonomy\Assignment 1\gs_extremelight.png",
]
paths_rgb = [
    r"C:\Users\Malik\OneDrive\Desktop\MASTERS\ECE 552 Visual Perception for Autonomy\Assignment 1\p2_extremedark_2.png",
    r"C:\Users\Malik\OneDrive\Desktop\MASTERS\ECE 552 Visual Perception for Autonomy\Assignment 1\fas.png",
    r"C:\Users\Malik\OneDrive\Desktop\MASTERS\ECE 552 Visual Perception for Autonomy\Assignment 1\p2_extremelight.png",
]
out_dir = r"C:\Users\Malik\OneDrive\Desktop\MASTERS\ECE 552 Visual Perception for Autonomy\Assignment 1\out"
os.makedirs(out_dir, exist_ok=True)

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

def ahe(img_u8, tile_h=128, tile_w=128, ksize=0):
    H, W = img_u8.shape
    Hp = int(np.ceil(H / tile_h) * tile_h)
    Wp = int(np.ceil(W / tile_w) * tile_w)
    padded = cv2.copyMakeBorder(img_u8, 0, Hp-H, 0, Wp-W, borderType=cv2.BORDER_REFLECT)
    out = np.zeros_like(padded, dtype=np.uint8)
    for y in range(0, Hp, tile_h):
        for x in range(0, Wp, tile_w):
            out[y:y+tile_h, x:x+tile_w] = he(padded[y:y+tile_h, x:x+tile_w])
    out = out[:H, :W]
    if ksize:
        if ksize % 2 == 0: ksize += 1
        out = cv2.GaussianBlur(out, (ksize, ksize), 0)
    return out

def stem(p): 
    return os.path.splitext(os.path.basename(p))[0]

def savefig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, name), dpi=300)
    plt.close()

def plot_rgb_hist(ax, hr, hg, hb, hI=None, labelI=None):
    x = np.arange(256)
    ax.plot(x, hr, color="red",   label="R")
    ax.plot(x, hg, color="green", label="G")
    ax.plot(x, hb, color="blue",  label="B")
    if hI is not None:
        ax.plot(x, hI, color="black", linewidth=2, label=labelI)
    ax.set_xlim(0,255)
    ax.legend()

#part1
for p in paths_gs:
    assert os.path.exists(p), p
    f = cv2.imread(p, cv2.IMREAD_GRAYSCALE); assert f is not None, p
    g = he(f)
    n  = np.bincount(f.ravel(), minlength=256)
    ng = np.bincount(g.ravel(), minlength=256)
    plt.figure(figsize=(11,6))
    plt.subplot(2,2,1); plt.imshow(f,cmap="gray"); plt.axis("off"); plt.title("Original")
    plt.subplot(2,2,2); plt.imshow(g,cmap="gray"); plt.axis("off"); plt.title("Gray Scale HE")
    plt.subplot(2,2,3); plt.plot(n);  plt.xlim(0,255); plt.title("Hist f(x,y)")
    plt.subplot(2,2,4); plt.plot(ng); plt.xlim(0,255); plt.title("Hist g(x,y)")
    savefig(f"p1_{stem(p)}_he.png")

#part 1 ahe
for p in paths_gs:
    assert os.path.exists(p), p
    f = cv2.imread(p, cv2.IMREAD_GRAYSCALE); assert f is not None, p
    g = ahe(f, tile_h=1024, tile_w=1024, ksize=5)
    n  = np.bincount(f.ravel(), minlength=256)
    ng = np.bincount(g.ravel(), minlength=256)
    plt.figure(figsize=(11,6))
    plt.subplot(2,2,1); plt.imshow(f,cmap="gray"); plt.axis("off"); plt.title("Original")
    plt.subplot(2,2,2); plt.imshow(g,cmap="gray"); plt.axis("off"); plt.title("Gray Scale AHE")
    plt.subplot(2,2,3); plt.plot(n);  plt.xlim(0,255); plt.title("Hist f(x,y)")
    plt.subplot(2,2,4); plt.plot(ng); plt.xlim(0,255); plt.title("Hist g(x,y)")
    savefig(f"p1_{stem(p)}_ahe.png")

# part 2
for p in paths_rgb:
    assert os.path.exists(p), p
    rgb8 = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB); assert rgb8 is not None, p
    rgb = rgb8.astype(np.float32) / 255.0
    R,G,B = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    I = (R+G+B)/3.0
    Iu = np.rint(I*255).clip(0,255).astype(np.uint8)
    Ip = he(Iu).astype(np.float32)/255.0
    ratio = Ip / (I + 1e-8)
    out = np.clip(np.stack([ratio*R, ratio*G, ratio*B], axis=2)*255.0, 0, 255).astype(np.uint8)

    hI  = np.bincount(Iu.ravel(), minlength=256)
    hIp = np.bincount(np.rint(Ip*255).clip(0,255).astype(np.uint8).ravel(), minlength=256)

    hr = np.bincount(rgb8[:,:,0].ravel(), minlength=256); hg = np.bincount(rgb8[:,:,1].ravel(), minlength=256); hb = np.bincount(rgb8[:,:,2].ravel(), minlength=256)
    hr2= np.bincount(out[:,:,0].ravel(), minlength=256);  hg2= np.bincount(out[:,:,1].ravel(), minlength=256);  hb2= np.bincount(out[:,:,2].ravel(), minlength=256)

    plt.figure(figsize=(12,6))
    plt.subplot(2,2,1); plt.imshow(rgb8); plt.axis("off"); plt.title("Original")
    plt.subplot(2,2,2); plt.imshow(out);  plt.axis("off"); plt.title("RGB Domain")

    plt.subplot(2,2,3); plot_rgb_hist(plt.gca(), hr, hg, hb, hI, "I");   plt.title("Original Histogram")
    plt.subplot(2,2,4); plot_rgb_hist(plt.gca(), hr2,hg2,hb2,hIp,"I'");  plt.title("Equalized Histogram'")

    savefig(f"p2_{stem(p)}_he.png")

#part 2 ahe
for p in paths_rgb:
    assert os.path.exists(p), p
    rgb8 = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB); assert rgb8 is not None, p
    rgb = rgb8.astype(np.float32) / 255.0
    R,G,B = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    I = (R+G+B)/3.0
    Iu = np.rint(I*255).clip(0,255).astype(np.uint8)
    Ip = ahe(Iu, tile_h=1024, tile_w=1024, ksize=5).astype(np.float32)/255.0
    ratio = Ip / (I + 1e-8)
    out = np.clip(np.stack([ratio*R, ratio*G, ratio*B], axis=2)*255.0, 0, 255).astype(np.uint8)

    hI  = np.bincount(Iu.ravel(), minlength=256)
    hIp = np.bincount(np.rint(Ip*255).clip(0,255).astype(np.uint8).ravel(), minlength=256)

    hr = np.bincount(rgb8[:,:,0].ravel(), minlength=256); hg = np.bincount(rgb8[:,:,1].ravel(), minlength=256); hb = np.bincount(rgb8[:,:,2].ravel(), minlength=256)
    hr2= np.bincount(out[:,:,0].ravel(), minlength=256);  hg2= np.bincount(out[:,:,1].ravel(), minlength=256);  hb2= np.bincount(out[:,:,2].ravel(), minlength=256)

    plt.figure(figsize=(12,6))
    plt.subplot(2,2,1); plt.imshow(rgb8); plt.axis("off"); plt.title("Original")
    plt.subplot(2,2,2); plt.imshow(out);  plt.axis("off"); plt.title("AHE RGB Domain")

    plt.subplot(2,2,3); plot_rgb_hist(plt.gca(), hr, hg, hb, hI, "I");   plt.title("Original Histogram")
    plt.subplot(2,2,4); plot_rgb_hist(plt.gca(), hr2,hg2,hb2,hIp,"I'");  plt.title("Equalized Histogram")

    savefig(f"p2_{stem(p)}_ahe.png")

# part 3
for p in paths_rgb:
    assert os.path.exists(p), p
    rgb8 = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB); assert rgb8 is not None, p
    rgb = rgb8.astype(np.float32) / 255.0
    R,G,B = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    eps = 1e-8

    I = (R+G+B)/3.0
    Cmin = np.minimum(np.minimum(R,G),B)
    S = 1.0 - (Cmin / (I + eps))

    num = 0.5*((R-G)+(R-B))
    den = np.sqrt((R-G)**2 + (R-B)*(G-B)) + eps
    th  = np.arccos(np.clip(num/den, -1.0, 1.0))
    H   = np.where(G >= B, th, 2*np.pi - th)

    Iu = np.rint(I*255).clip(0,255).astype(np.uint8)
    Ip = he(Iu).astype(np.float32)/255.0

    hI  = np.bincount(Iu.ravel(), minlength=256)
    hIp = np.bincount(np.rint(Ip*255).clip(0,255).astype(np.uint8).ravel(), minlength=256)

    h = H; s = np.clip(S,0,1); i = np.clip(Ip,0,1)
    R2 = np.zeros_like(i); G2 = np.zeros_like(i); B2 = np.zeros_like(i)
    m0 = (h < 2*np.pi/3)
    m1 = (h >= 2*np.pi/3) & (h < 4*np.pi/3)
    m2 = (h >= 4*np.pi/3)

    h0 = h
    B2[m0] = i[m0]*(1-s[m0])
    R2[m0] = i[m0]*(1 + s[m0]*np.cos(h0[m0])/(np.cos(np.pi/3 - h0[m0]) + eps))
    G2[m0] = 3*i[m0] - (R2[m0] + B2[m0])

    h1 = h - 2*np.pi/3
    R2[m1] = i[m1]*(1-s[m1])
    G2[m1] = i[m1]*(1 + s[m1]*np.cos(h1[m1])/(np.cos(np.pi/3 - h1[m1]) + eps))
    B2[m1] = 3*i[m1] - (R2[m1] + G2[m1])

    h2 = h - 4*np.pi/3
    G2[m2] = i[m2]*(1-s[m2])
    B2[m2] = i[m2]*(1 + s[m2]*np.cos(h2[m2])/(np.cos(np.pi/3 - h2[m2]) + eps))
    R2[m2] = 3*i[m2] - (G2[m2] + B2[m2])

    out = np.clip(np.stack([R2,G2,B2], axis=2)*255.0, 0, 255).astype(np.uint8)

    hr = np.bincount(rgb8[:,:,0].ravel(),minlength=256); hg = np.bincount(rgb8[:,:,1].ravel(),minlength=256); hb = np.bincount(rgb8[:,:,2].ravel(),minlength=256)
    hr2= np.bincount(out[:,:,0].ravel(),minlength=256);  hg2= np.bincount(out[:,:,1].ravel(),minlength=256);  hb2= np.bincount(out[:,:,2].ravel(),minlength=256)

    plt.figure(figsize=(12,6))
    plt.subplot(2,2,1); plt.imshow(rgb8); plt.axis("off"); plt.title("Original")
    plt.subplot(2,2,2); plt.imshow(out);  plt.axis("off"); plt.title("HSI")

    plt.subplot(2,2,3); plot_rgb_hist(plt.gca(), hr, hg, hb, hI, "I");   plt.title("Original Histogram")
    plt.subplot(2,2,4); plot_rgb_hist(plt.gca(), hr2,hg2,hb2,hIp,"I'");  plt.title("Equalized Histogram")

    savefig(f"p3_{stem(p)}_he.png")

# part 3 ahe
for p in paths_rgb:
    assert os.path.exists(p), p
    rgb8 = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB); assert rgb8 is not None, p
    rgb = rgb8.astype(np.float32) / 255.0
    R,G,B = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    eps = 1e-8

    I = (R+G+B)/3.0
    Cmin = np.minimum(np.minimum(R,G),B)
    S = 1.0 - (Cmin / (I + eps))

    num = 0.5*((R-G)+(R-B))
    den = np.sqrt((R-G)**2 + (R-B)*(G-B)) + eps
    th  = np.arccos(np.clip(num/den, -1.0, 1.0))
    H   = np.where(G >= B, th, 2*np.pi - th)

    Iu = np.rint(I*255).clip(0,255).astype(np.uint8)
    Ip = ahe(Iu, tile_h=1024, tile_w=1024, ksize=5).astype(np.float32)/255.0

    hI  = np.bincount(Iu.ravel(), minlength=256)
    hIp = np.bincount(np.rint(Ip*255).clip(0,255).astype(np.uint8).ravel(), minlength=256)

    h = H; s = np.clip(S,0,1); i = np.clip(Ip,0,1)
    R2 = np.zeros_like(i); G2 = np.zeros_like(i); B2 = np.zeros_like(i)
    m0 = (h < 2*np.pi/3)
    m1 = (h >= 2*np.pi/3) & (h < 4*np.pi/3)
    m2 = (h >= 4*np.pi/3)

    h0 = h
    B2[m0] = i[m0]*(1-s[m0])
    R2[m0] = i[m0]*(1 + s[m0]*np.cos(h0[m0])/(np.cos(np.pi/3 - h0[m0]) + eps))
    G2[m0] = 3*i[m0] - (R2[m0] + B2[m0])

    h1 = h - 2*np.pi/3
    R2[m1] = i[m1]*(1-s[m1])
    G2[m1] = i[m1]*(1 + s[m1]*np.cos(h1[m1])/(np.cos(np.pi/3 - h1[m1]) + eps))
    B2[m1] = 3*i[m1] - (R2[m1] + G2[m1])

    h2 = h - 4*np.pi/3
    G2[m2] = i[m2]*(1-s[m2])
    B2[m2] = i[m2]*(1 + s[m2]*np.cos(h2[m2])/(np.cos(np.pi/3 - h2[m2]) + eps))
    R2[m2] = 3*i[m2] - (G2[m2] + B2[m2])

    out = np.clip(np.stack([R2,G2,B2], axis=2)*255.0, 0, 255).astype(np.uint8)

    hr = np.bincount(rgb8[:,:,0].ravel(),minlength=256); hg = np.bincount(rgb8[:,:,1].ravel(),minlength=256); hb = np.bincount(rgb8[:,:,2].ravel(),minlength=256)
    hr2= np.bincount(out[:,:,0].ravel(),minlength=256);  hg2= np.bincount(out[:,:,1].ravel(),minlength=256);  hb2= np.bincount(out[:,:,2].ravel(),minlength=256)

    plt.figure(figsize=(12,6))
    plt.subplot(2,2,1); plt.imshow(rgb8); plt.axis("off"); plt.title("Original")
    plt.subplot(2,2,2); plt.imshow(out);  plt.axis("off"); plt.title("AHE HSI")

    plt.subplot(2,2,3); plot_rgb_hist(plt.gca(), hr, hg, hb, hI, "I");   plt.title("Original Histogram")
    plt.subplot(2,2,4); plot_rgb_hist(plt.gca(), hr2,hg2,hb2,hIp,"I'");  plt.title("Equalized Histogram")

    savefig(f"p3_{stem(p)}_ahe.png")

# part 4
for p in paths_rgb:
    assert os.path.exists(p), p
    rgb8 = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB); assert rgb8 is not None, p
    hsv = cv2.cvtColor(rgb8, cv2.COLOR_RGB2HSV)
    H,S,V = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    Vp = he(V)
    out = cv2.cvtColor(np.dstack([H,S,Vp]).astype(np.uint8), cv2.COLOR_HSV2RGB)

    hV  = np.bincount(V.ravel(), minlength=256)
    hVp = np.bincount(Vp.ravel(), minlength=256)

    hr = np.bincount(rgb8[:,:,0].ravel(),minlength=256); hg = np.bincount(rgb8[:,:,1].ravel(),minlength=256); hb = np.bincount(rgb8[:,:,2].ravel(),minlength=256)
    hr2= np.bincount(out[:,:,0].ravel(),minlength=256);  hg2= np.bincount(out[:,:,1].ravel(),minlength=256);  hb2= np.bincount(out[:,:,2].ravel(),minlength=256)

    plt.figure(figsize=(12,6))
    plt.subplot(2,2,1); plt.imshow(rgb8); plt.axis("off"); plt.title("Original")
    plt.subplot(2,2,2); plt.imshow(out);  plt.axis("off"); plt.title("HSV")

    plt.subplot(2,2,3); plot_rgb_hist(plt.gca(), hr, hg, hb, hV, "V");    plt.title("Original Histogram")
    plt.subplot(2,2,4); plot_rgb_hist(plt.gca(), hr2,hg2,hb2,hVp,"V'");   plt.title("Equalized Histogram")

    savefig(f"p4_{stem(p)}_he.png")

# part 4 ahe
for p in paths_rgb:
    assert os.path.exists(p), p
    rgb8 = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB); assert rgb8 is not None, p
    hsv = cv2.cvtColor(rgb8, cv2.COLOR_RGB2HSV)
    H,S,V = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    Vp = ahe(V, tile_h=1024, tile_w=1024, ksize=5)
    out = cv2.cvtColor(np.dstack([H,S,Vp]).astype(np.uint8), cv2.COLOR_HSV2RGB)

    hV  = np.bincount(V.ravel(), minlength=256)
    hVp = np.bincount(Vp.ravel(), minlength=256)

    hr = np.bincount(rgb8[:,:,0].ravel(),minlength=256); hg = np.bincount(rgb8[:,:,1].ravel(),minlength=256); hb = np.bincount(rgb8[:,:,2].ravel(),minlength=256)
    hr2= np.bincount(out[:,:,0].ravel(),minlength=256);  hg2= np.bincount(out[:,:,1].ravel(),minlength=256);  hb2= np.bincount(out[:,:,2].ravel(),minlength=256)

    plt.figure(figsize=(12,6))
    plt.subplot(2,2,1); plt.imshow(rgb8); plt.axis("off"); plt.title("Original")
    plt.subplot(2,2,2); plt.imshow(out);  plt.axis("off"); plt.title("AHE HSV")

    plt.subplot(2,2,3); plot_rgb_hist(plt.gca(), hr, hg, hb, hV, "V");    plt.title("Original Histogram")
    plt.subplot(2,2,4); plot_rgb_hist(plt.gca(), hr2,hg2,hb2,hVp,"V'");   plt.title("Equalized Histogram'")

    savefig(f"p4_{stem(p)}_ahe.png")
