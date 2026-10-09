# Robot simulator setup

Two isolated Python 3.10 venvs. Nothing was installed system-wide and the system Python 3.12 is untouched.

| venv | Path | Key packages |
|---|---|---|
| `robot-sim` | `~/.venvs/robot-sim` | robosuite 1.5.2, mujoco 3.3.7, numpy 1.26.4 |
| `libero` | `~/.venvs/libero` | LIBERO (editable, `./LIBERO`), robosuite 1.4.0, mujoco 2.3.7, numpy 1.22.4, torch 2.2.2+cpu |

LIBERO needs its own venv because it is written against robosuite 1.4 and does not run on 1.5.

## Rerun the tests

```bash
cd ~/Documents/"humanoid project"
unset PYTHONPATH        # your shell profile adds ROS Jazzy (Python 3.12) packages; keep them out of these venvs

# robosuite: Panda + cube (Lift), 100 steps -> test_frame.png, test_video.mp4
source ~/.venvs/robot-sim/bin/activate
python tests/smoke/robosuite_smoke.py
deactivate

# LIBERO: libero_object task 0, 50 steps -> libero_frame.png
source ~/.venvs/libero/bin/activate
python tests/smoke/libero_smoke.py
deactivate
```

Expected output from `tests/smoke/robosuite_smoke.py`: the cube pose at start and end, then `saved ... (mean pixel ~215-220)` and `OK`. A mean pixel far from 215-220 means a garbled render (see below).

## Rendering

- Both scripts default to `MUJOCO_GL=egl`, which renders headless on the RTX 2080. No window or display is needed.
- To skip rendering entirely: `HEADLESS_STATE_ONLY=1 python tests/smoke/robosuite_smoke.py`.
- `MUJOCO_GL=osmesa` (CPU rendering) is **not** available, because it needs `sudo apt install libosmesa6`.
- The exit-time `EGLError` tracebacks ("Exception ignored in ... __del__") are harmless teardown noise.
- One render out of about 8 came out garbled: the first run right after switching mujoco versions. It did not reproduce. If you see it, rerun.

## Version pins (and why)

- **mujoco < 3.4 with robosuite 1.5.2.** robosuite only requires `mujoco>=3.3`, so pip installs the newest (3.14), which crashes on `env.reset()` with an `AssertionError` in `get_joint_qpos_addr`. In mujoco 3.14, `jnt_type in (mjtJoint.mjJNT_HINGE, ...)` evaluates to False.
- **LIBERO venv:** these follow LIBERO's own `requirements.txt` (robosuite 1.4.0, numpy 1.22.4, bddl 1.0.1, gym 0.25.2). torch is pinned below 2.6 because 2.6 changed `torch.load` to `weights_only=True`, which breaks loading LIBERO's init-state files. Training-only packages (wandb, transformers, robomimic, hydra) are not installed.
- **LIBERO must be installed with `--config-settings editable_mode=compat`.** Its top-level `libero/` folder has no `__init__.py`, so a normal `pip install -e .` installs nothing importable.
- LIBERO's config lives at `~/.libero/config.yaml` and points into `./LIBERO`. Demo datasets are not downloaded; the "datasets path does not exist" warning is expected.

## Recreate from scratch

```bash
unset PYTHONPATH
# uv supplies Python 3.10 without sudo (stored in ~/.local/share/uv/python)
python3 -m venv ~/.venvs/uvboot && ~/.venvs/uvboot/bin/pip install uv
UV=~/.venvs/uvboot/bin/uv
$UV python install 3.10

# robot-sim
$UV venv --python 3.10 --seed ~/.venvs/robot-sim
~/.venvs/robot-sim/bin/pip install robosuite==1.5.2 "mujoco>=3.3,<3.4" imageio imageio-ffmpeg

# libero
$UV venv --python 3.10 --seed ~/.venvs/libero
~/.venvs/libero/bin/pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
~/.venvs/libero/bin/pip install robosuite==1.4.0 mujoco==2.3.7 numpy==1.22.4 bddl==1.0.1 easydict==1.9 \
    gym==0.25.2 cloudpickle==2.1.0 future==0.18.2 matplotlib==3.5.3 opencv-python==4.6.0.66 pyyaml termcolor tqdm imageio
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git   # tested at commit 8f1084e
(cd LIBERO && ~/.venvs/libero/bin/pip install -e . --no-deps --config-settings editable_mode=compat)
echo n | ~/.venvs/libero/bin/python -c "import libero.libero"     # writes ~/.libero/config.yaml non-interactively
```

## Additions for the cube project (5 Oct, evening)

### `robot-sim` (the main environment)
- **New packages:** matplotlib, and CPU-only torch 2.14.1 for the world model.
- **Install everything:**
  ```bash
  ~/.venvs/robot-sim/bin/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
  ```
- **Tests:**
  ```bash
  python -m pytest tests -q
  ```

### `lerobot` (VLA stretch goal only)
- **Location:** `~/.venvs/lerobot`, Python 3.12.
- **Why separate:** lerobot 0.6.1 needs Python >= 3.12 and numpy >= 2.
- **Contents:**
  - `lerobot[smolvla,dataset]==0.6.1`, with torch 2.11 CUDA wheels; it sees the RTX 2080.
  - robosuite 1.5.2 with mujoco 3.3.7 and numpy 2.2.6, so `src.eval_vla` can run a SmolVLA checkpoint in the same process as the sim.
- **Known conflict:** pip reports one conflict, `mink` (robosuite's optional humanoid IK) wanting numpy < 2. The Panda path does not use mink.
- **Rebuild:**
  ```bash
  python3 -m venv ~/.venvs/uvboot && ~/.venvs/uvboot/bin/pip install uv
  ~/.venvs/uvboot/bin/uv venv --python 3.12 --seed ~/.venvs/lerobot
  ~/.venvs/lerobot/bin/pip install "lerobot[smolvla,dataset]==0.6.1"
  ~/.venvs/lerobot/bin/pip install robosuite==1.5.2 "mujoco>=3.3,<3.4" opencv-python scipy imageio imageio-ffmpeg matplotlib
  ~/.venvs/lerobot/bin/pip install numpy==2.2.6   # robosuite's dependencies pull numpy down to 1.26, which breaks lerobot
  ```
