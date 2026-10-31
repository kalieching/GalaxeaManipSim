# xmem: prompt G0.5 in GalaxeaManipSim

Type a prompt in a browser → the [G0.5 VLA](https://github.com/OpenGalaxea/GalaxeaVLA) predicts actions → an R1 Lite or R1 Pro carries them out in this simulator.

| File | Purpose |
|---|---|
| `xmem/vla_client.py` | Steps the sim with G0.5's actions; serves the web page (live view, prompt, Run/Stop) |
| `xmem/serve_policy_5090.py` | Starts the G0.5 policy server on RTX 5090 (see Notes) |
| `xmem/fetch_assets.py` | Downloads all object meshes (replaces the dead Google Drive link) |
| `galaxea_sim/envs/robotwin/cube_bowl_place.py` | Scene: red 4 cm cube + bowl; success = cube released inside the bowl. `R1LiteCubeBowlPlace-v0`, `R1ProCubeBowlPlace-v0` |

Clone both repos side by side:

```
xmem/
├── GalaxeaManipSim/   # this fork, branch xmem/cube-bowl
└── GalaxeaVLA/        # OpenGalaxea/GalaxeaVLA @ 89f2322
```

## 1. Simulator

```bash
git clone -b xmem/cube-bowl https://github.com/kalieching/GalaxeaManipSim.git
cd GalaxeaManipSim
conda create -n galaxea-sim python=3.10 -y && conda activate galaxea-sim
pip install -e .
pip install "numpy<2" "setuptools<81" websockets msgpack
python xmem/fetch_assets.py   # ~2 min, ~420 MB into galaxea_sim/assets/robotwin_models
```

## 2. VLA

```bash
git clone https://github.com/OpenGalaxea/GalaxeaVLA.git
cd GalaxeaVLA && git checkout 89f2322
uv tool install cmake   # only if cmake is not installed
CMAKE_POLICY_VERSION_MINIMUM=3.5 uv sync --index-strategy unsafe-best-match
```

Accept the license at <https://huggingface.co/OpenGalaxea/G05> (gated, auto-approved) with your own HF account, then (~12 GB):

```bash
hf auth login   # paste a read token from huggingface.co/settings/tokens; check with `hf auth whoami`
hf download OpenGalaxea/G05 --local-dir checkpoints \
  --include "g05-base/*" --include "action_tokenizer.pt" --include "qwen3_5_2b_base_processor/*"
```

## 3. Run

Terminal 1, in `GalaxeaVLA`, on a free GPU (~12 GB); ready at `Policy server listening` after 1–2 min:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:scripts .venv/bin/python ../GalaxeaManipSim/xmem/serve_policy_5090.py \
  --ckpt_path checkpoints/g05-base/checkpoints/model_state_dict.pt --host 127.0.0.1 --port 8765 \
  eval_embodiment=galaxea_r1lite model.model_arch.attn_implementation=sdpa
```

Terminal 2, in `GalaxeaManipSim` with `galaxea-sim` active:

```bash
python xmem/vla_client.py --robot r1lite
```

Open <http://localhost:8000>. From a laptop, forward port 8000 first: VS Code Remote-SSH → Ports panel → Forward a Port → `8000`, or on the laptop run `ssh -L 8000:127.0.0.1:8000 <user>@<gpu-host>`.

| Page control | Action |
|---|---|
| Run | New scene, then run the prompt until success (cube released in the bowl) or `--max-steps` (default 900 = 60 s sim time) |
| Stop | Stop the current run |
| New scene | Re-randomize cube and bowl |

View: scene camera (left), robot head camera (right; this and the wrist cameras are what G0.5 sees).

**R1 Pro:** `eval_embodiment=galaxea_r1pro` in terminal 1 and `--robot r1pro` in terminal 2. Both must match.

## Notes

- **Zero-shot.** `g05-base` was not trained on this sim. On R1 Lite it has picked up the cube and carried it to the bowl, but not yet released it.
- **Speed.** ~8 sim steps/s on an RTX 5090 (sim and robot run at 15 Hz), so playback is slower than real time.
- **RTX 5090.** FlashAttention-4 fails there (`Operation creation failed`), and G0.5's vision encoder uses it whenever it is installed. `serve_policy_5090.py` forces its PyTorch SDPA fallback. Where FA4 works (H100/B200), `scripts/serve_policy.py` with the same arguments should work unchanged (untested).
- **Units.** Arm joints: radians in both. Gripper: sim finger travel 0–0.05 m ↔ G0.5 0 (closed)–100 (open). Parts G0.5 does not predict hold their last command.
- **R1 (original)** has no G0.5 embodiment; only R1 Lite and R1 Pro are supported.
- **R1 Lite placement.** Its head camera sees only the far part of the table, so its cube and bowl are placed as close to the robot as possible while staying fully in that view. The env has no scripted solution.
- **Pins.** `numpy<2`: mplib 0.2.1 segfaults on numpy 2. `setuptools<81`: sapien 3.0.0b1 imports `pkg_resources`.
- **`uv.lock`.** `uv sync` rewrites its index URLs (aliyun mirror → pypi.org; same versions). Don't commit it.
- **License.** G0.5 is non-commercial and forbids exposing it to third parties. Keep both servers on `127.0.0.1`.
