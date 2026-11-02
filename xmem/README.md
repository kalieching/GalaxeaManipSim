# xmem: prompt G0.5 in GalaxeaManipSim

Type a prompt in a browser → the [G0.5 VLA](https://github.com/OpenGalaxea/GalaxeaVLA) predicts actions → an R1 Lite or R1 Pro carries them out in this simulator.

| File | Purpose |
|---|---|
| `xmem/vla_client.py` | Steps the sim with G0.5's actions; serves the web page (live scene/head/wrist views, prompt, Run/Stop) |
| `xmem/serve_policy_5090.py` | Starts the G0.5 policy server on RTX 5090 (see Notes) |
| `galaxea_sim/scripts/fetch_assets.py` | Downloads all object meshes (replaces the dead Google Drive link) |
| `galaxea_sim/envs/robotwin/cube_bowl_place.py` | Scene: red 4 cm cube + bowl; success = cube released inside the bowl. `R1LiteCubeBowlPlace-v0`, `R1ProCubeBowlPlace-v0` |

Clone both repos side by side:

```
xmem/
├── GalaxeaManipSim/   # this fork (kalieching/GalaxeaManipSim)
└── GalaxeaVLA/        # OpenGalaxea/GalaxeaVLA @ 89f2322
```

## 1. Simulator

```bash
git clone https://github.com/kalieching/GalaxeaManipSim.git
cd GalaxeaManipSim
conda create -n galaxea-sim python=3.10 -y && conda activate galaxea-sim
pip install -e .
pip install "numpy<2" "setuptools<81" websockets msgpack
python -m galaxea_sim.scripts.fetch_assets   # ~2 min, ~420 MB into galaxea_sim/assets/robotwin_models
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

Terminal 1, in `GalaxeaVLA`, on a free GPU (~15 GB while running; one server per 32 GB GPU); ready at `Policy server listening` after 1–2 min:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:scripts .venv/bin/python ../GalaxeaManipSim/xmem/serve_policy_5090.py \
  --ckpt_path checkpoints/g05-base/checkpoints/model_state_dict.pt --host 127.0.0.1 --port 8765 \
  eval_embodiment=galaxea_r1lite model.model_arch.attn_implementation=sdpa \
  model.model_arch.discrete_action=true model.model_arch.continuous_action=false
```

Terminal 2, in `GalaxeaManipSim` with `galaxea-sim` active:

```bash
python xmem/vla_client.py --robot r1lite
```

Open <http://localhost:8000>. From a laptop, forward port 8000 first: VS Code Remote-SSH → Ports panel → Forward a Port → `8000`, or on the laptop run `ssh -L 8000:127.0.0.1:8000 <user>@<gpu-host>`.

| Page control | Action |
|---|---|
| Run | Put the robot, cube and bowl back to the current scene's start, then run the prompt until success (cube released in the bowl) or `--max-steps` (default 1200 = 80 s sim time) |
| Stop | Stop the current run |
| New scene | New random cube and bowl layout (robot at its start pose) |

View: scene camera and head camera (top), left and right wrist cameras (bottom). The head and wrist cameras are what G0.5 sees.

**R1 Pro:** `eval_embodiment=galaxea_r1pro` in terminal 1 and `--robot r1pro` in terminal 2. Both must match.

## Notes

- **Zero-shot success** (`g05-base` was not trained on this sim; 2026-11-02, 1200-step limit, prompt "Pick up the cube from the tabletop surface and place it in the bowl.", before the stiff-arm and cube-placement changes): R1 Lite 2/8, R1 Pro 3/17. With the later changes and the "red cube" prompt, R1 Pro scored 0–1/10 on a fixed 10-scene set for every combination of arm drives and cube placement tested, i.e. G0.5's zero-shot rate is low (~5–10%) either way (not yet measured with the corrected finger box below). Failed runs mostly push the cube or hover without grasping. Naming the color ("red cube", now the default prompt) helped in manual runs.
- **Discrete actions.** The server flags `discrete_action=true continuous_action=false` (as in Galaxea's R1 Pro command) decode actions from G0.5's action tokens. R1 Pro needs them; with the default flags it scored 0/12. R1 Lite: 2/8 vs 1/8 default.
- **R1 Pro start pose.** `R1ProCubeBowlPlace-v0` starts its arms at the mean pose of G0.5's real R1 Pro data (`R1PRO_G05_INIT_QPOS`). From upstream's `R1PRO_INIT_QPOS`, which is outside that data, G0.5 barely moves.
- **R1 Pro finger collision (fixed in `galaxea_sim/assets/r1_pro/robot.urdf`).** R1 Pro's finger mesh is R1 Lite's rotated 90°, but upstream reused R1 Lite's collision box unrotated, so it stuck out sideways: grasped objects sat ~12 mm (up to 26 mm) from the visible fingers. The box is now rotated to match (`0.04 0.01 0.06`), giving R1 Pro the same finger collision as R1 Lite (~6 mm average gap where the finger's inner surface is recessed). Upstream R1 Pro scripted tasks are unchanged (BlocksStackEasy 6/8, DualBottlesPickEasy 7/8 with either box).
- **Arm drives.** `CubeBowlPlaceEnv` stiffens the arm joints (1e5/4e3 instead of 1000/200) so joints reach each 15 Hz target in 1–2 steps instead of ~10. Gripper drives are unchanged.
- **Speed.** Playback runs at roughly half real time on an RTX 5090 (~8 sim steps/s measured with two views; sim and robot run at 15 Hz): ~40 ms per step to render the four live views, plus ~0.45 s of G0.5 inference every 16 steps.
- **RTX 5090.** FlashAttention-4 fails there (`Operation creation failed`), and G0.5's vision encoder uses it whenever it is installed. `serve_policy_5090.py` forces its PyTorch SDPA fallback. Where FA4 works (H100/B200), `scripts/serve_policy.py` with the same arguments should work unchanged (untested).
- **Units.** Arm joints: radians in both. Gripper: sim finger travel 0–0.05 m ↔ G0.5 0 (closed)–100 (open). Parts G0.5 does not predict hold their last command.
- **R1 (original)** has no G0.5 embodiment; only R1 Lite and R1 Pro are supported.
- **Placement.** R1 Lite's head camera sees only the far part of the table, so its cube and bowl are placed as close to the robot as possible while staying fully in that view. R1 Pro's cube sits ~5 cm from its starting grippers. On both, the cube is re-placed if a gripper hides it from the head camera. The env has no scripted solution.
- **Pins.** `numpy<2`: mplib 0.2.1 segfaults on numpy 2. `setuptools<81`: sapien 3.0.0b1 imports `pkg_resources`.
- **`uv.lock`.** `uv sync` rewrites its index URLs (aliyun mirror → pypi.org; same versions). Don't commit it.
- **`address already in use`.** An old server or client is still running: `pkill -f serve_policy_5090.py` or `pkill -f xmem/vla_client.py`, then start it again. After a network change, re-forward port 8000 (VS Code Ports panel: Stop Forwarding, then Forward a Port).
- **License.** G0.5 is non-commercial and forbids exposing it to third parties. Keep both servers on `127.0.0.1`.
