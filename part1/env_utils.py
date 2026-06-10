from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def import_gym():
    try:
        import gymnasium as gym

        return gym, "gymnasium"
    except ImportError:
        import gym

        return gym, "gym"


def make_env(env_id: str, seed: int | None = None, render_mode: str | None = None, max_episode_steps: int | None = None):
    gym, backend = import_gym()
    kwargs: dict[str, Any] = {}
    if render_mode is not None:
        kwargs["render_mode"] = render_mode

    env = gym.make(env_id, **kwargs)

    if max_episode_steps is not None:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)

    if seed is not None:
        try:
            env.action_space.seed(seed)
        except AttributeError:
            pass
        try:
            env.observation_space.seed(seed)
        except AttributeError:
            pass

    return env, backend


def reset_env(env, seed: int | None = None):
    if seed is not None:
        try:
            out = env.reset(seed=seed)
        except TypeError:
            try:
                env.seed(seed)
            except AttributeError:
                pass
            out = env.reset()
    else:
        out = env.reset()

    if isinstance(out, tuple) and len(out) == 2:
        return out
    return out, {}


def step_env(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
        info = dict(info)
        info["terminated"] = bool(terminated)
        info["truncated"] = bool(truncated)
        return obs, float(reward), done, info

    obs, reward, done, info = out
    info = dict(info)
    info["terminated"] = bool(done)
    info["truncated"] = False
    return obs, float(reward), bool(done), info


def unwrap_env(env):
    current = env
    visited = set()
    while hasattr(current, "env") and id(current) not in visited:
        visited.add(id(current))
        current = current.env
    return getattr(current, "unwrapped", current)


def get_mujoco_model(env):
    unwrapped = unwrap_env(env)
    if hasattr(unwrapped, "model"):
        return unwrapped.model
    if hasattr(unwrapped, "sim") and hasattr(unwrapped.sim, "model"):
        return unwrapped.sim.model
    return None


def _decode_name(name):
    if isinstance(name, bytes):
        return name.decode("utf-8")
    return str(name)


@dataclass
class EnvDescription:
    observation_space: str
    action_space: str
    body_names: list[str]
    body_masses: list[float]
    body_dofnum: list[int]
    nv: int | None
    nu: int | None


def describe_env(env) -> EnvDescription:
    model = get_mujoco_model(env)
    body_names: list[str] = []
    body_masses: list[float] = []
    body_dofnum: list[int] = []
    nv = None
    nu = None

    if model is not None:
        if hasattr(model, "body_names"):
            body_names = [_decode_name(name) for name in model.body_names]
        elif hasattr(model, "nbody") and hasattr(model, "body"):
            body_names = [_decode_name(model.body(i).name) for i in range(model.nbody)]

        if hasattr(model, "body_mass"):
            body_masses = [float(x) for x in np.asarray(model.body_mass).reshape(-1)]
        if hasattr(model, "body_dofnum"):
            body_dofnum = [int(x) for x in np.asarray(model.body_dofnum).reshape(-1)]
        if hasattr(model, "nv"):
            nv = int(model.nv)
        if hasattr(model, "nu"):
            nu = int(model.nu)

    return EnvDescription(
        observation_space=str(env.observation_space),
        action_space=str(env.action_space),
        body_names=body_names,
        body_masses=body_masses,
        body_dofnum=body_dofnum,
        nv=nv,
        nu=nu,
    )


def assert_continuous_box_space(env):
    gym, _ = import_gym()
    if not isinstance(env.observation_space, gym.spaces.Box):
        raise TypeError(f"Expected a continuous Box observation space, got {env.observation_space!r}")
    if not isinstance(env.action_space, gym.spaces.Box):
        raise TypeError(f"Expected a continuous Box action space, got {env.action_space!r}")

