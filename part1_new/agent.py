from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_hidden_sizes(value: str | Iterable[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        return tuple(int(item.strip()) for item in value.split(",") if item.strip())
    return tuple(int(item) for item in value)


def build_mlp(input_dim: int, hidden_sizes: tuple[int, ...], output_dim: int, activation=nn.Tanh) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous_dim = input_dim
    for hidden_dim in hidden_sizes:
        layers.append(nn.Linear(previous_dim, hidden_dim))
        layers.append(activation())
        previous_dim = hidden_dim
    layers.append(nn.Linear(previous_dim, output_dim))
    return nn.Sequential(*layers)


def discounted_returns(rewards: list[float], gamma: float) -> torch.Tensor:
    values: list[float] = []
    running = 0.0
    for reward in reversed(rewards):
        running = float(reward) + gamma * running
        values.append(running)
    values.reverse()
    return torch.tensor(values, dtype=torch.float32)


@dataclass
class UpdateStats:
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    total_loss: float = 0.0


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: tuple[int, ...],
        action_low: np.ndarray,
        action_high: np.ndarray,
        log_std_init: float = -0.5,
    ):
        super().__init__()
        self.net = build_mlp(obs_dim, hidden_sizes, act_dim)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std_init)))

        low = torch.as_tensor(action_low, dtype=torch.float32)
        high = torch.as_tensor(action_high, dtype=torch.float32)
        finite = torch.isfinite(low) & torch.isfinite(high)
        low = torch.where(finite, low, torch.full_like(low, -1.0))
        high = torch.where(finite, high, torch.full_like(high, 1.0))
        self.register_buffer("action_low", low)
        self.register_buffer("action_high", high)
        self.register_buffer("action_scale", (high - low) / 2.0)
        self.register_buffer("action_bias", (high + low) / 2.0)

    def distribution(self, obs: torch.Tensor) -> Normal:
        raw_mean = self.net(obs)
        mean = torch.tanh(raw_mean) * self.action_scale + self.action_bias
        log_std = torch.clamp(self.log_std, -5.0, 2.0)
        std = torch.exp(log_std).expand_as(mean)
        return Normal(mean, std)

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        dist = self.distribution(obs)
        raw_action = dist.mean if deterministic else dist.sample()
        action = torch.max(torch.min(raw_action, self.action_high), self.action_low)
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action.squeeze(0), log_prob.squeeze(0), entropy.squeeze(0)


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden_sizes: tuple[int, ...]):
        super().__init__()
        self.net = build_mlp(obs_dim, hidden_sizes, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class ReinforceAgent:
    name = "reinforce"

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        hidden_sizes: tuple[int, ...] = (64, 64),
        lr: float = 3e-4,
        gamma: float = 0.99,
        baseline: float = 0.0,
        entropy_coef: float = 0.0,
        max_grad_norm: float = 1.0,
        normalize_advantages: bool = False,
        reward_to_go: bool = True,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.baseline = baseline
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.normalize_advantages = normalize_advantages
        self.reward_to_go = reward_to_go
        self.policy = GaussianPolicy(obs_dim, act_dim, hidden_sizes, action_low, action_high).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.log_probs: list[torch.Tensor] = []
        self.entropies: list[torch.Tensor] = []
        self.rewards: list[float] = []

    def select_action(self, obs: np.ndarray, deterministic: bool = False):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if deterministic:
            with torch.no_grad():
                action, _, _ = self.policy.act(obs_tensor, deterministic=True)
            return action.cpu().numpy(), {}
        action, log_prob, entropy = self.policy.act(obs_tensor, deterministic=False)
        return action.detach().cpu().numpy(), {"log_prob": log_prob, "entropy": entropy}

    def record(self, obs, action_info, reward: float, next_obs, done: bool) -> None:
        self.log_probs.append(action_info["log_prob"])
        self.entropies.append(action_info["entropy"])
        self.rewards.append(float(reward))

    def update(self) -> UpdateStats:
        if not self.rewards:
            return UpdateStats()

        if self.reward_to_go:
            returns = discounted_returns(self.rewards, self.gamma).to(self.device)
        else:
            full_return = discounted_returns(self.rewards, self.gamma)[0].item()
            returns = torch.full((len(self.rewards),), full_return, dtype=torch.float32, device=self.device)

        advantages = returns - float(self.baseline)
        if self.normalize_advantages and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        log_probs = torch.stack(self.log_probs)
        entropies = torch.stack(self.entropies)
        policy_loss = -(log_probs * advantages.detach()).sum()
        entropy = entropies.mean()
        loss = policy_loss - self.entropy_coef * entropies.sum()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        stats = UpdateStats(
            policy_loss=float(policy_loss.detach().cpu()),
            value_loss=0.0,
            entropy=float(entropy.detach().cpu()),
            total_loss=float(loss.detach().cpu()),
        )
        self.clear_episode()
        return stats

    def clear_episode(self) -> None:
        self.log_probs.clear()
        self.entropies.clear()
        self.rewards.clear()

    def save(self, path: str | Path, metadata: dict | None = None) -> None:
        payload = {
            "agent_type": self.name,
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "metadata": metadata or {},
        }
        torch.save(payload, path)

    def load(self, path: str | Path) -> dict:
        payload = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(payload["policy"])
        if "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        return payload.get("metadata", {})


class ActorCriticAgent:
    name = "actor_critic"

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        hidden_sizes: tuple[int, ...] = (64, 64),
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        entropy_coef: float = 0.0,
        value_coef: float = 0.5,
        max_grad_norm: float = 1.0,
        target_mode: str = "td",
        normalize_advantages: bool = False,
        device: str = "cpu",
    ):
        if target_mode not in {"td", "mc"}:
            raise ValueError("target_mode must be either 'td' or 'mc'")

        self.device = torch.device(device)
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.target_mode = target_mode
        self.normalize_advantages = normalize_advantages

        self.policy = GaussianPolicy(obs_dim, act_dim, hidden_sizes, action_low, action_high).to(self.device)
        self.value = ValueNetwork(obs_dim, hidden_sizes).to(self.device)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.parameters(), "lr": actor_lr},
                {"params": self.value.parameters(), "lr": critic_lr},
            ]
        )

        self.log_probs: list[torch.Tensor] = []
        self.entropies: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []
        self.rewards: list[float] = []
        self.next_observations: list[np.ndarray] = []
        self.dones: list[bool] = []

    def select_action(self, obs: np.ndarray, deterministic: bool = False):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if deterministic:
            with torch.no_grad():
                action, _, _ = self.policy.act(obs_tensor, deterministic=True)
            return action.cpu().numpy(), {}

        action, log_prob, entropy = self.policy.act(obs_tensor, deterministic=False)
        value = self.value(obs_tensor)
        return action.detach().cpu().numpy(), {"log_prob": log_prob, "entropy": entropy, "value": value}

    def record(self, obs, action_info, reward: float, next_obs, done: bool) -> None:
        self.log_probs.append(action_info["log_prob"])
        self.entropies.append(action_info["entropy"])
        self.values.append(action_info["value"])
        self.rewards.append(float(reward))
        self.next_observations.append(np.asarray(next_obs, dtype=np.float32))
        self.dones.append(bool(done))

    def _targets(self) -> torch.Tensor:
        rewards = torch.as_tensor(self.rewards, dtype=torch.float32, device=self.device)
        if self.target_mode == "mc":
            return discounted_returns(self.rewards, self.gamma).to(self.device)

        next_obs = torch.as_tensor(np.asarray(self.next_observations), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(self.dones, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            next_values = self.value(next_obs)
        return rewards + self.gamma * next_values * (1.0 - dones)

    def update(self) -> UpdateStats:
        if not self.rewards:
            return UpdateStats()

        values = torch.stack(self.values)
        log_probs = torch.stack(self.log_probs)
        entropies = torch.stack(self.entropies)
        targets = self._targets()
        advantages = targets - values
        if self.normalize_advantages and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = F.mse_loss(values, targets.detach())
        entropy = entropies.mean()
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(list(self.policy.parameters()) + list(self.value.parameters()), self.max_grad_norm)
        self.optimizer.step()

        stats = UpdateStats(
            policy_loss=float(policy_loss.detach().cpu()),
            value_loss=float(value_loss.detach().cpu()),
            entropy=float(entropy.detach().cpu()),
            total_loss=float(loss.detach().cpu()),
        )
        self.clear_episode()
        return stats

    def clear_episode(self) -> None:
        self.log_probs.clear()
        self.entropies.clear()
        self.values.clear()
        self.rewards.clear()
        self.next_observations.clear()
        self.dones.clear()

    def save(self, path: str | Path, metadata: dict | None = None) -> None:
        payload = {
            "agent_type": self.name,
            "policy": self.policy.state_dict(),
            "value": self.value.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "metadata": metadata or {},
        }
        torch.save(payload, path)

    def load(self, path: str | Path) -> dict:
        payload = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(payload["policy"])
        if "value" in payload:
            self.value.load_state_dict(payload["value"])
        if "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        return payload.get("metadata", {})

