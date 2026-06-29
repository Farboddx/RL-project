# Part 2 Setup

Use a separate Python environment for Part 2. Do not reuse the Part 1 Python 3.13 environment.

Reason:

- `panda-gym` depends on `pybullet`.
- `pybullet` does not install cleanly from source on the current Python 3.13 setup.
- `panda-gym` also requires `numpy<2`.

Recommended setup:

```bash
cd /Users/hamidreza/Documents/Codex/2026-05-29/files-mentioned-by-the-user-report/faiml_rl_26_source
python3.11 -m venv .venv-part2
source .venv-part2/bin/activate
python -m pip install --upgrade pip
python -m pip install -r part2/requirements_part2.txt
cd part2/panda-gym
python -m pip install -e .
cd ..
```

If `python3.11` is not found, install Python 3.11 first. On macOS, the simplest options are:

- install Python 3.11 from `python.org`;
- or install Miniconda and create a Python 3.11 environment.

After setup, test:

```bash
python inspect_push_env.py
```

Expected masses:

- source: `1 kg`
- target: `5 kg`

Quick sanity training:

```bash
python train_sb3.py --algo sac --env-type source --sampling-strategy none --timesteps 5000 --seed 0
```

Full training matrix:

```bash
bash run_part2_matrix.sh 500000 0 sac
```

The verified local 5,000 timestep run used the prepared Conda Python 3.11 environment:

```bash
cd /Users/hamidreza/Documents/Codex/2026-05-29/files-mentioned-by-the-user-report/faiml_rl_26_source/part2
/Users/hamidreza/Documents/Codex/2026-05-29/files-mentioned-by-the-user-report/.conda-part2-py311/bin/python inspect_push_env.py
PATH=/Users/hamidreza/Documents/Codex/2026-05-29/files-mentioned-by-the-user-report/.conda-part2-py311/bin:$PATH bash run_part2_matrix.sh 5000 0 sac
```
