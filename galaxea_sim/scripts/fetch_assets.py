"""Download the RoboTwin object meshes into galaxea_sim/assets/robotwin_models and robotwin2_models.

Replaces the dead Google Drive link for robotwin_models.zip:
  - RoboTwin 1.0 objects.zip (latest)  -> all objects
  - RoboTwin 1.0 objects.zip @ f1ee6a6  -> 020_hammer_2, tools (removed from later uploads)
  - RoboTwin 2.0 objects.zip            -> 002_bowl variant 1 (CubeBowlPlace)
  - RoboTwin 2.0 objects.zip            -> the Kitting containers and objects, into robotwin2_models (same layout as
                                           RoboTwin 2.0: <name>/visual/base<i>.glb, collision/base<i>.glb,
                                           model_data<i>.json), with the handle cut off basket 110_basket variant 1
Members of the older and the 2.0 zip are fetched with HTTP range requests, so only they are downloaded.
Steps whose files are already there are skipped.
"""
import io
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import trimesh
import tyro

from galaxea_sim import ASSETS_DIR
from galaxea_sim.envs.robotwin.kitting import BASKET_HANDLE, KITTING_MODELS

HF = "https://huggingface.co/datasets/TianxingChen"
ROBOTWIN1_LATEST = f"{HF}/RoboTwin/resolve/main/objects.zip"
ROBOTWIN1_F1EE6A6 = f"{HF}/RoboTwin/resolve/f1ee6a6a14af204f23fd0372320b7845990eecd6/objects.zip"
ROBOTWIN2 = f"{HF}/RoboTwin2.0/resolve/main/objects.zip"
BOWL_FILES = {
    "objects/002_bowl/visual/base1.glb": "002_bowl/base1.glb",
    "objects/002_bowl/collision/base1.glb": "002_bowl/collision_base1.glb",
    "objects/002_bowl/model_data1.json": "002_bowl/model_data1.json",
}


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file, so zipfile can read single members."""

    def __init__(self, url):
        self.url, self.pos = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
            self.size = int(r.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def read(self, n=-1):
        n = self.size - self.pos if n < 0 else min(n, self.size - self.pos)
        if n <= 0:
            return b""
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        with urllib.request.urlopen(req) as r:
            data = r.read()
        self.pos += len(data)
        return data


def extract_members(url: str, dest: Path, mapping: dict):
    with zipfile.ZipFile(HttpRangeFile(url)) as z:
        for src, rel in mapping.items():
            out = dest / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(z.read(src))


def strip_handle(model_dir: Path, variant: int, cut: float):
    """Remove a basket's handle: the connected mesh pieces and convex collision pieces that reach above `cut` (m,
    after scaling, the handle sits above the rim with a clear gap). model_data's extents and center follow the mesh."""
    data_path = model_dir / f"model_data{variant}.json"
    data = json.loads(data_path.read_text())
    cut = cut / data["scale"][1]  # model frame, y up
    mesh = trimesh.load(model_dir / "visual" / f"base{variant}.glb", force="mesh")
    labels = trimesh.graph.connected_component_labels(mesh.face_adjacency, node_count=len(mesh.faces))
    top = np.full(labels.max() + 1, -np.inf)
    np.maximum.at(top, labels, mesh.vertices[mesh.faces][:, :, 1].max(axis=1))
    mesh.update_faces(top[labels] <= cut)
    mesh.remove_unreferenced_vertices()
    mesh.export(model_dir / "visual" / f"base{variant}.glb")
    pieces = trimesh.load(model_dir / "collision" / f"base{variant}.glb")
    kept = trimesh.Scene()
    for node in pieces.graph.nodes_geometry:
        transform, geometry = pieces.graph[node]
        piece = pieces.geometry[geometry].copy()
        piece.apply_transform(transform)
        if piece.bounds[1][1] <= cut:
            kept.add_geometry(piece, node_name=node)
    kept.export(model_dir / "collision" / f"base{variant}.glb")
    lo, hi = mesh.bounds
    data.update(extents=(hi - lo).tolist(), center=((hi + lo) / 2).tolist())
    data_path.write_text(json.dumps(data))


def main(dest: Path = ASSETS_DIR / "robotwin_models", dest2: Path = ASSETS_DIR / "robotwin2_models"):
    dest.mkdir(parents=True, exist_ok=True)
    if not (dest / "002_bowl" / "base1.glb").exists():
        fetch_robotwin_models(dest)
    kitting = {f"{name}/{sub}/base{i}.glb" for name, i in KITTING_MODELS for sub in ("visual", "collision")} | \
              {f"{name}/model_data{i}.json" for name, i in KITTING_MODELS}
    missing = sorted(f for f in kitting if not (dest2 / f).exists())
    if missing:
        print(f"[4/4] Kitting objects from RoboTwin 2.0 -> {dest2}")
        extract_members(ROBOTWIN2, dest2, {f"objects/{f}": f for f in missing})
        name, variant, cut = BASKET_HANDLE
        if f"{name}/model_data{variant}.json" in missing:
            strip_handle(dest2 / name, variant, cut)
    print("done")


def fetch_robotwin_models(dest: Path):
    print(f"[1/3] RoboTwin 1.0 objects.zip (~300 MB) -> {dest}")
    with tempfile.TemporaryDirectory(dir=dest) as tmp:
        zip_path = Path(tmp) / "objects.zip"
        with urllib.request.urlopen(ROBOTWIN1_LATEST) as r, open(zip_path, "wb") as f:
            shutil.copyfileobj(r, f)
        with zipfile.ZipFile(zip_path) as z:
            for name in z.namelist():
                if name.startswith("objects/") and not name.endswith("/"):
                    out = dest / name[len("objects/"):]
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(z.read(name))

    print("[2/3] 020_hammer_2, tools from RoboTwin 1.0 @ f1ee6a6")
    with zipfile.ZipFile(HttpRangeFile(ROBOTWIN1_F1EE6A6)) as z:
        old = [n for n in z.namelist()
               if n.startswith(("objects/020_hammer_2/", "objects/tools/")) and not n.endswith("/")]
    extract_members(ROBOTWIN1_F1EE6A6, dest, {n: n[len("objects/"):] for n in old})

    print("[3/3] 002_bowl from RoboTwin 2.0")
    extract_members(ROBOTWIN2, dest, BOWL_FILES)


if __name__ == "__main__":
    tyro.cli(main)
