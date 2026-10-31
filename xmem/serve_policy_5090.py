"""Run GalaxeaVLA's scripts/serve_policy.py with the vision encoder on PyTorch SDPA.

FlashAttention-4 (flash_attn.cute) fails to build kernels on RTX 5090 (sm_120) with
"Operation creation failed", and qwen35/vision.py uses it whenever it imports, ignoring
attn_implementation. Clearing its hook makes vision.py take its SDPA fallback.

Run from GalaxeaVLA with the same arguments as serve_policy.py (see xmem/README.md):
    cd GalaxeaVLA
    PYTHONPATH=src:scripts .venv/bin/python ../GalaxeaManipSim/xmem/serve_policy_5090.py \
        --ckpt_path checkpoints/g05-base/checkpoints/model_state_dict.pt --host 127.0.0.1 --port 8765 \
        eval_embodiment=galaxea_r1lite model.model_arch.attn_implementation=sdpa
"""
import g05.models.g05.qwen35.vision as vision

vision._flash_attn_varlen = None

import serve_policy  # noqa: E402

if __name__ == "__main__":
    serve_policy.main()
