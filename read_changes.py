from PIL import Image
import numpy as np, pandas as pd

idx = np.load("db/stego/00001_changes_idx.npy")
print(idx.shape, idx[:20])

h, w = np.array(Image.open("db/stego/00001_stego.png").convert("L")).shape

r = idx // w
c = idx %  w
coords = np.c_[r, c]
print(coords[:10])

np.savetxt("img_changes_coords.csv", coords, fmt="%d", delimiter=",",
           header="row,col", comments="")

coords = pd.read_csv("img_changes_coords.csv").to_numpy()
H, W = 512, 512
mask = np.zeros((H,W), dtype=bool)
mask[coords[:,0], coords[:,1]] = True
rate = mask.mean()
print(f'rate={rate}')

cover = np.array(Image.open("db/cover/00001.tif").convert("L"), np.int16)
stego = np.array(Image.open("db/stego/00001_stego.png").convert("L"), np.int16)
delta = stego - cover
print(f'delta={delta}')
n_p1 = np.count_nonzero(delta[mask] == +1)
n_m1 = np.count_nonzero(delta[mask] == -1)