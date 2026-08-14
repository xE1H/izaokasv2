# Training Runbook — vast.ai GPU from scratch

How to rent a GPU, install the stack, train, watch TensorBoard, benchmark, and
shut down — the exact steps that worked, including the gotchas. Everything here
runs from a local machine (a Mac is fine — it only drives the remote over SSH;
Isaac Sim itself cannot run on macOS).

Related: [Track A](#track-a--the-optimized-controller-pipeline) below, for the
deterministic-controller pipeline.

---

## 0. What you need

- A **vast.ai** account with credit, and the CLI: `uv tool install vastai`
  (or `pip install vastai`), then `vastai set api-key <KEY>`.
- An **SSH keypair** (`~/.ssh/id_ed25519`). `ssh-keygen -t ed25519` if you
  don't have one.
- A **GitHub read token** for cloning the private fork on the remote (stored in
  `.env`, gitignored).

---

## 1. Pick an instance

Isaac Sim 5.1 needs an **NVIDIA driver in the 550–580 series** (590+ crashes in
`librtx`). An **RTX 4090** is the sweet spot (24 GB, fast, ~$0.30/hr on the
marketplace). Filter offers to a compatible driver, an EU host, and enough disk:

```bash
vastai search offers \
  'verified=true rentable=true gpu_name=RTX_4090 num_gpus=1 \
   driver_version >= 550.00.00 driver_version < 590.00.00 \
   disk_space > 90' \
  -o 'dph_total'          # cheapest first
```

Read the **NV Driver** column (it must be 550.x–580.x — the web UI hides it,
the CLI shows it), plus reliability (R > 98%) and network speed (Isaac Sim is a
~10 GB download). Note the **offer id** (first column). Offer ids are
ephemeral — grab a fresh one right before creating.

---

## 2. Create and connect

⚠️ **Gotcha (team accounts):** `vastai attach ssh` silently fails to inject the
key on team accounts, so **bake the public key into the onstart script** and it
Just Works. Also, instances sometimes come up **stopped** — you must `start`
them.

```bash
PUBKEY="$(cat ~/.ssh/id_ed25519.pub)"
ONSTART="mkdir -p /root/.ssh && echo '$PUBKEY' >> /root/.ssh/authorized_keys && \
chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys && \
apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -y \
git curl build-essential ca-certificates && touch /root/onstart.done"

vastai create instance <OFFER_ID> \
  --image nvidia/cuda:12.5.1-devel-ubuntu24.04 \
  --disk 60 --ssh --direct --onstart-cmd "$ONSTART"
# (60 GB is plenty: Isaac Sim install ~30 GB. Don't over-allocate — disk is
#  billed even while stopped, $0.20/GB/month.)
# NB: the "-cudnn-devel-" image tag does NOT exist for 12.5.1; use plain -devel.

vastai start instance <CONTRACT_ID>      # if it comes up "stopped"
vastai ssh-url <CONTRACT_ID>             # -> ssh://root@sshN.vast.ai:PORT
```

Add an SSH alias so later commands are short (fill in host/port from `ssh-url`):

```
# ~/.ssh/config
Host vast4090
    HostName sshN.vast.ai
    Port <PORT>
    User root
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    ServerAliveInterval 30
```

```bash
ssh vast4090 'nvidia-smi --query-gpu=name,driver_version --format=csv,noheader'
# confirm it's an RTX 4090 with driver 550–580 BEFORE installing anything
```

Tip: run `caffeinate -s` locally (macOS) so your laptop sleeping doesn't drop
SSH. The training itself runs under `nohup` on the remote and survives
disconnects regardless.

---

## 3. Install the stack

```bash
# on the remote
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# clone the private fork with the read token, then scrub it from the URL
git clone "https://x-access-token:${GITHUB_TOKEN}@github.com/xE1H/izaokas.git" \
  LituanicaX_IsaacSimChallenge
cd LituanicaX_IsaacSimChallenge
printf 'GITHUB_TOKEN=%s\n' "$GITHUB_TOKEN" > .env && chmod 600 .env
git remote set-url origin https://github.com/xE1H/izaokas.git
git config credential.helper 'store --file=/root/.git-credentials'
printf 'https://x-access-token:%s@github.com\n' "$GITHUB_TOKEN" > /root/.git-credentials
```

⚠️ **Gotcha (install.sh):** the fork commits IsaacLab as plain files (not a real
submodule) but keeps a stale `.gitmodules`, so `install.sh` crashes at the
`git checkout v2.3.0` step. The IsaacLab files are already the correct v2.3.0
and `pyproject.toml` installs from the local path — **skip the submodule lines**
and run the rest by hand:

```bash
uv venv --python 3.11
uv sync                                     # the big ~10 GB Isaac Sim download
uv pip install --force-reinstall "opencv-python" "numpy<2.0.0"
uv pip install "setuptools<82.0.0"
```

Extra system libs needed for **headless rendering / video** (not for plain
training):

```bash
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  libgl1 libglx-mesa0 libegl1 libgles2 libxrandr2 libxinerama1 libxcursor1 \
  libxi6 libxext6 libx11-6 libxrender1 libgomp1 libvulkan1 vulkan-tools \
  mesa-vulkan-drivers libglib2.0-0t64 libsm6 libice6 libxt6 libglu1-mesa
```

⚠️ **`libglu1-mesa` is not optional for video.** Without `libGLU.so.1` the RTX
material plugin (`rtx.neuraylib`) fails to load and every frame renders **all
black** — the run still "succeeds" and writes an mp4, just a black one. Confirm
Vulkan sees the GPU with `vulkaninfo --summary | grep deviceName` (should list
`NVIDIA GeForce RTX 4090`, not just `llvmpipe`).

Verify:

```bash
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
.venv/bin/python -m pytest tests/ -q      # ~200 CPU tests, ~2 min
```

---

## 4. Train

Isaac Sim prompts for a EULA on first run; set `OMNI_KIT_ACCEPT_EULA=YES` so it
doesn't hang in a non-interactive shell. Run under `nohup` so it survives SSH
drops:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
cd ~/LituanicaX_IsaacSimChallenge

# fresh run (num_envs default is 3072; use 1024 for faster feedback)
nohup uv run python -u -m teamcode.train --num_envs 1024 --headless \
  agent.max_iterations=500 > ~/train.log 2>&1 &

# continue from the latest checkpoint (reward/hyperparam changes OK; do NOT
# change observation width or network size — those break resume)
nohup uv run python -u -m teamcode.train --resume --num_envs 1024 --headless \
  agent.max_iterations=50 > ~/train.log 2>&1 &
```

- Checkpoints land in `logs/<timestamp>/model_<iter>.pt`, saved every 25 iters.
- ~1 min/iter at 1024 envs, ~2 min/iter at 3072.
- `--resume` loads the **newest** checkpoint under `logs/`. If a stray partial
  run created a newer one, delete that run folder first so it resumes from the
  right model.

---

## 5. Watch TensorBoard

The trainer starts TensorBoard on the remote (port 6006). Tunnel to it:

```bash
ssh -N -f -L 6006:localhost:6006 vast4090     # background tunnel
# open http://localhost:6006
pkill -f "6006:localhost:6006"                 # close it later
```

Quick text status instead (from the repo root, uses the `vast4090` alias):

```bash
ssh vast4090 'tail -5 ~/train.log'      # iteration, reward, episode length
ssh vast4090 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv'
```

Metrics that matter: `Train/mean_reward` (should climb), `Mean episode length`
(cars surviving; ~1800 = full episode), `Lap/best_lap_time_s` (appears once a
car completes a lap). **`best_lap_time_s` during training is noisy** (best of N
cars with exploration) — always confirm real speed with `benchmark`.

---

## 6. Benchmark and submit

`benchmark` is the official score: 10 cars, one deterministic lap each from the
origin, fastest wins. It **publishes to the leaderboard by default**.

```bash
export OMNI_KIT_ACCEPT_EULA=YES
export LITUANICAX_TEAM="Vilnius Lyceum Carbotics"

# score WITHOUT publishing (to compare models safely)
uv run python -u -m lituanicax_sdk.benchmark --headless --no-submit \
  --checkpoint logs/<run>/model_<iter>.pt

# score AND publish the improved entry
uv run python -u -m lituanicax_sdk.benchmark --headless \
  --checkpoint logs/<run>/model_<iter>.pt
```

It uploads a policy bundle (checkpoint + teamcode) so the organisers can re-run
and verify the lap. The lap shows "awaiting verification" until they do.
Result is also written to `submission.json` next to the checkpoint.

## 6b. Render a lap video (optional)

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run python -u -m teamcode.play --headless \
  --num_envs 1 --checkpoint logs/<run>/model_<iter>.pt --video
# writes logs/<run>/videos/play/rl-video-step-0.mp4  (needs the render libs from step 3)
```

---

## 7. Back up, then stop or destroy

Checkpoints are tiny (<1 MB). Pull the good ones off the (marketplace) host:

```bash
scp vast4090:'~/LituanicaX_IsaacSimChallenge/logs/<run>/model_<iter>.pt' checkpoints-backup/
```

Then:

- **`vastai stop instance <ID>`** — halts GPU billing, keeps the disk (~$0.02/hr
  for 60 GB). Restart with `vastai start instance <ID>`; everything's still
  there. Use for short breaks (hours–days).
- **`vastai destroy instance <ID>`** — deletes everything, $0 after. Use for
  long breaks; re-run this runbook (~20 min) to rebuild. Confirm with
  `echo y | vastai destroy instance <ID>`.

> Marketplace hosts can vanish, so a "stopped" instance is ~99% safe, not 100%.
> Keep checkpoints backed up locally and code/progress on GitHub, and destroying
> costs you nothing.

---

## Track A — the optimized controller pipeline

A deterministic controller, optimized against the simulator with CMA-ES, used to
measure what this track allows and then to teach a policy. `tools/` and `teacher/`
in this repo; see the module docstrings for why each piece is shaped as it is.

Half of it needs no simulator and is unit-tested anywhere:

```bash
# on any machine, including a Mac (needs torch, scipy, pytest — no Isaac Sim)
python -m pytest tests/ -q             # 351 tests
python -m teacher.warmstart --report   # the starting parameters, and a verdict
```

The other half needs a GPU box built as in steps 1-3 above, plus:

```bash
uv pip install cma      # teacher.optimize needs it; nothing else does
```

Then, in order. **Step 1 is a go/no-go — do not skip it.**

```bash
export OMNI_KIT_ACCEPT_EULA=YES
cd ~/LituanicaX_IsaacSimChallenge

# 1. Measure the car. Exits 3 if the tightest corner cannot be steered at all,
#    in which case a pure-pursuit teacher will not complete a lap and the reason
#    is geometry, not gains.
uv run python -m tools.probe --headless

# 2. How reproducible is the simulator? Decides whether a candidate's score
#    needs averaging over repeats — worth knowing before a day of GPU time.
uv run python -m tools.determinism --headless --envs 64 --out artifacts/det-64.npz
uv run python -m tools.determinism --headless --envs 1  --out artifacts/det-1.npz
uv run python -m tools.determinism --compare artifacts/det-64.npz artifacts/det-1.npz
uv run python -m tools.determinism --headless --restore

# 3. Warm start, now against measured numbers rather than guesses.
uv run python -m teacher.warmstart --report

# 4. Search. Start at 64 candidates until a generation's wall-clock is known;
#    raise it once you can afford to.
nohup uv run python -u -m teacher.optimize --headless \
  --population 64 --generations 100 > ~/optimize.log 2>&1 &

# 5. The reportable number, at the 60 s window a real attempt gets.
uv run python -m teacher.optimize --headless --measure artifacts/teacher.json
```

`T_teacher` from step 5 is the point of the whole exercise. Read it against the
board's best: comfortably under it means there is something to distil, level with
it means the controller has only matched what RL already does.

⚠️ **The four Isaac-dependent modules have never been run** — Isaac Sim does not
run on macOS. Expect to fix import-order and API-signature mistakes in
`tools/harness.py`, `tools/probe.py`, `tools/determinism.py` and
`teacher/optimize.py` on the first session, before trusting any measurement.

⚠️ **Never run this with `--device cpu` and more than one car.** Isaac Lab only
filters collisions between environments on the GPU, so on CPU every car shares
one 0.70 m corridor and crashes into the others. `make_env` refuses unless you
pass `allow_cpu=True`.

---

## Gotcha checklist (all bit us at least once)

| Symptom | Cause / fix |
|---|---|
| `Permission denied (publickey)` forever | Team account — bake pubkey into `--onstart-cmd`, don't rely on `attach ssh` |
| Instance stuck "stopped/created" | `vastai start instance <ID>` |
| Image `manifest not found` | `-cudnn-devel-` tag absent for 12.5.1 — use plain `-devel` |
| `install.sh` dies at `git checkout v2.3.0` | Fake IsaacLab submodule — skip submodule lines, run the rest by hand (step 3) |
| Training hangs at "Do you accept the EULA?" | `export OMNI_KIT_ACCEPT_EULA=YES` |
| Video render: `libXt.so.6 not found` | Install the render libs (step 3) |
| Video renders **all black** (mp4 tiny, ~50 KB) | `libGLU.so.1` missing — `apt-get install -y libglu1-mesa`, then re-render |
| `stop`ped instance won't restart ("GPU in use, hours–weeks") | Host GPU taken. Disk is safe — use the **web panel copy button** to copy the data dir to a new running instance (CLI `vastai copy` needs the source running, hangs on a stopped one). `scp` checkpoints off before stopping. |
| Driver ≥ 590 | Isaac Sim crashes in `librtx` — pick a 550–580 host |