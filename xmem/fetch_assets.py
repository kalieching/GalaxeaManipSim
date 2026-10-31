"""Download the object meshes GalaxeaManipSim tasks need into galaxea_sim/assets/robotwin_models.

Replaces the dead Google Drive link in the upstream README:
  - RoboTwin 1.0 objects.zip (latest)          -> all objects
  - RoboTwin 1.0 objects.zip @ f1ee6a6          -> 020_hammer_2, tools (removed upstream later)
  - RoboTwin 2.0 objects.zip, 002_bowl variant 1 -> 002_bowl (CubeBowlPlace)
Only the needed members of the two older/larger zips are fetched, via HTTP range requests.
"""
import argparse
import io
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

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


def extract_members(url, dest, mapping):
    with zipfile.ZipFile(HttpRangeFile(url)) as z:
        for src, rel in mapping.items():
            out = dest / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(z.read(src))


def main():
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dest", type=Path, default=repo / "galaxea_sim/assets/robotwin_models")
    dest = ap.parse_args().dest
    dest.mkdir(parents=True, exist_ok=True)

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
    print("done")


if __name__ == "__main__":
    main()
