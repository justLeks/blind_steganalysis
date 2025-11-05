from pathlib import Path
from PIL import Image
import numpy as np
import conseal as cl

def to_u8_gray(p):
    im = Image.open(p).convert("L")
    arr = np.array(im)
    if arr.dtype == np.uint16:
        arr = (arr / 257).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8, copy=False)
    return arr

def embed_and_log(src_dir, dst_dir, method="HUGO", alpha=0.4, seed=12345):
    src_dir, dst_dir = (
        (Path.cwd() / src_dir).resolve(),
        (Path.cwd() / dst_dir).resolve()
    )
    files = [p for p in src_dir.rglob("*")]
    for p in files:
        x0 = to_u8_gray(p)
        if method.upper()=="HUGO":
            x1 = cl.hugo.simulate_single_channel(x0=x0, alpha=alpha, seed=seed)
        elif method.upper()=="MIPOD":
            x1 = cl.mipod.simulate_single_channel(x0=x0, alpha=alpha, seed=seed)
        else:
            raise ValueError("Supported: HUGO, MiPOD")

        mask = x1 != x0
        idx  = np.flatnonzero(mask)
        out_dir = dst_dir / p.relative_to(src_dir).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(x1).save(out_dir / (p.stem + "_stego.png"))
        np.save(out_dir / (p.stem + "_changes_idx.npy"), idx)

if __name__ == "__main__":
    embed_and_log("db/cover", "db/stego")
