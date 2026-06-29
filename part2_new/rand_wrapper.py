from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np


@dataclass
class ADRState:
    mass_min: float
    mass_max: float
    last_mass: float | None = None
    last_sample_type: str = "none"
    episodes: int = 0
    expansions: int = 0
    contractions: int = 0


class RandomizationWrapper(gym.Wrapper):
    """Randomize the PandaPush object mass at every episode.

    Modes:
    - none: keep the environment's native mass.
    - udr: sample uniformly from a fixed mass range.
    - adr: adapt the min/max mass range using boundary performance.

    The official Part 2 environment has source mass 1 kg and target mass 5 kg.
    UDR/ADR should be applied only during source-domain training.
    """

    def __init__(
        self,
        env: gym.Env,
        mass_range: tuple[float, float] = (0.5, 8.0),
        mode: str = "none",
        adr_initial_range: tuple[float, float] = (1.0, 2.0),
        adr_step: float = 0.25,
        adr_boundary_prob: float = 0.5,
        adr_success_threshold: float = 0.5,
        verbose: bool = False,
    ):
        super().__init__(env)
        if mode not in {"none", "udr", "adr"}:
            raise ValueError("mode must be one of: none, udr, adr")

        global_min, global_max = sorted(float(x) for x in mass_range)
        init_min, init_max = sorted(float(x) for x in adr_initial_range)
        init_min = float(np.clip(init_min, global_min, global_max))
        init_max = float(np.clip(init_max, global_min, global_max))
        if init_min == init_max:
            init_max = min(global_max, init_min + adr_step)

        self.mode = mode
        self.mass_range = (global_min, global_max)
        self.mass_min_limit = global_min
        self.mass_max_limit = global_max
        self.adr_step = float(adr_step)
        self.adr_boundary_prob = float(np.clip(adr_boundary_prob, 0.0, 1.0))
        self.adr_success_threshold = float(adr_success_threshold)
        self.verbose = verbose
        self.rng = np.random.default_rng()

        self.adr = ADRState(mass_min=init_min, mass_max=init_max)
        self.current_mass: float | None = None
        self.last_sample_type = "none"
        self.episode_return = 0.0
        self.episode_successes: list[float] = []

    @property
    def mass_min(self) -> float:
        return self.adr.mass_min if self.mode == "adr" else self.mass_min_limit

    @property
    def mass_max(self) -> float:
        return self.adr.mass_max if self.mode == "adr" else self.mass_max_limit

    def _update_adr_from_previous_episode(self) -> None:
        if self.mode != "adr" or self.adr.last_mass is None:
            return
        if self.adr.last_sample_type not in {"lower", "upper"}:
            return

        success_value = max(self.episode_successes) if self.episode_successes else 0.0
        success = success_value >= self.adr_success_threshold
        old_min, old_max = self.adr.mass_min, self.adr.mass_max

        if self.adr.last_sample_type == "lower":
            if success:
                self.adr.mass_min = max(self.mass_min_limit, self.adr.mass_min - self.adr_step)
                self.adr.expansions += int(self.adr.mass_min != old_min)
            else:
                self.adr.mass_min = min(self.adr.mass_max - 1e-6, self.adr.mass_min + self.adr_step)
                self.adr.contractions += int(self.adr.mass_min != old_min)
        else:
            if success:
                self.adr.mass_max = min(self.mass_max_limit, self.adr.mass_max + self.adr_step)
                self.adr.expansions += int(self.adr.mass_max != old_max)
            else:
                self.adr.mass_max = max(self.adr.mass_min + 1e-6, self.adr.mass_max - self.adr_step)
                self.adr.contractions += int(self.adr.mass_max != old_max)

        if self.verbose and (old_min, old_max) != (self.adr.mass_min, self.adr.mass_max):
            print(
                "[adr-update] "
                f"success={success_value:.2f} "
                f"boundary={self.adr.last_sample_type} "
                f"range=[{self.adr.mass_min:.2f}, {self.adr.mass_max:.2f}]"
            )

    def _sample_mass(self) -> tuple[float | None, str]:
        if self.mode == "none":
            return None, "none"

        if self.mode == "udr":
            return float(self.rng.uniform(self.mass_min_limit, self.mass_max_limit)), "uniform"

        if self.rng.random() < self.adr_boundary_prob:
            if self.rng.random() < 0.5:
                return float(self.adr.mass_min), "lower"
            return float(self.adr.mass_max), "upper"
        return float(self.rng.uniform(self.adr.mass_min, self.adr.mass_max)), "interior"

    def _set_object_mass(self, mass: float) -> None:
        task = self.env.unwrapped.task
        sim = task.sim
        object_body_id = sim._bodies_idx["object"]
        sim.physics_client.changeDynamics(
            bodyUniqueId=object_body_id,
            linkIndex=-1,
            mass=float(mass),
        )
        task.current_mass = float(mass)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_return += float(reward)
        if isinstance(info, dict) and "is_success" in info:
            self.episode_successes.append(float(info["is_success"]))
        if isinstance(info, dict):
            info = dict(info)
            info["mass"] = self.current_mass
            info["mass_sample_type"] = self.last_sample_type
            if self.mode == "adr":
                info["adr_mass_min"] = self.adr.mass_min
                info["adr_mass_max"] = self.adr.mass_max
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._update_adr_from_previous_episode()

        new_mass, sample_type = self._sample_mass()
        if new_mass is not None:
            self._set_object_mass(new_mass)

        self.current_mass = new_mass
        self.last_sample_type = sample_type
        self.adr.last_mass = new_mass
        self.adr.last_sample_type = sample_type
        self.adr.episodes += 1
        self.episode_return = 0.0
        self.episode_successes = []

        if self.verbose and new_mass is not None:
            print(
                f"[{self.mode}] mass={new_mass:.2f} "
                f"range=[{self.mass_min:.2f}, {self.mass_max:.2f}] "
                f"type={self.last_sample_type}"
            )

        obs, info = self.env.reset(**kwargs)
        if isinstance(info, dict):
            info = dict(info)
            info["mass"] = self.current_mass
            info["mass_sample_type"] = self.last_sample_type
        return obs, info

