from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

ALGORITHM_NAMES: Tuple[str, ...] = ("PPO", "A2C", "DQN")


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - float(np.max(logits))
    exp = np.exp(shifted)
    denom = float(np.sum(exp))
    if denom <= 0:
        return np.full_like(logits, 1.0 / len(logits), dtype=np.float64)
    return exp / denom


@dataclass
class ActionDecision:
    action: int
    action_prob: float
    aux: Dict[str, float] = field(default_factory=dict)


class StrategyLearner(ABC):
    def __init__(
        self,
        name: str,
        state_dim: int,
        n_actions: int,
        seed: int,
    ) -> None:
        self.name = name.upper()
        self.state_dim = int(state_dim)
        self.n_actions = int(n_actions)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

    @abstractmethod
    def act(self, state: np.ndarray, explore: bool) -> ActionDecision:
        raise NotImplementedError

    @abstractmethod
    def learn(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        decision: ActionDecision,
    ) -> None:
        raise NotImplementedError


class DQNLearner(StrategyLearner):
    def __init__(self, state_dim: int, n_actions: int, seed: int) -> None:
        super().__init__(name="DQN", state_dim=state_dim, n_actions=n_actions, seed=seed)
        self.gamma = 0.96
        self.lr = 0.035
        self.eps_start = 0.22
        self.eps_end = 0.02
        self.eps_decay = 450.0
        self.step_count = 0

        self.q_w = self.rng.normal(0.0, 0.06, size=(self.n_actions, self.state_dim))
        self.q_b = self.rng.normal(0.0, 0.01, size=(self.n_actions,))

    def _epsilon(self) -> float:
        return float(
            self.eps_end
            + (self.eps_start - self.eps_end) * np.exp(-float(self.step_count) / self.eps_decay)
        )

    def _q_values(self, state: np.ndarray) -> np.ndarray:
        return self.q_w @ state + self.q_b

    def act(self, state: np.ndarray, explore: bool) -> ActionDecision:
        self.step_count += 1
        q_values = self._q_values(state)
        greedy_action = int(np.argmax(q_values))
        epsilon = self._epsilon() if explore else 0.0

        if explore and self.rng.random() < epsilon:
            action = int(self.rng.integers(0, self.n_actions))
        else:
            action = greedy_action

        if epsilon <= 0:
            action_prob = 1.0 if action == greedy_action else 0.0
        elif action == greedy_action:
            action_prob = (1.0 - epsilon) + epsilon / self.n_actions
        else:
            action_prob = epsilon / self.n_actions

        return ActionDecision(action=action, action_prob=float(action_prob), aux={"epsilon": epsilon})

    def learn(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        decision: ActionDecision,
    ) -> None:
        del decision
        q_values = self._q_values(state)
        q_sa = float(q_values[action])
        q_next = self._q_values(next_state)
        target = float(reward)
        if not done:
            target += self.gamma * float(np.max(q_next))

        td_error = target - q_sa
        self.q_w[action] += self.lr * td_error * state
        self.q_b[action] += self.lr * td_error


class A2CLearner(StrategyLearner):
    def __init__(self, state_dim: int, n_actions: int, seed: int) -> None:
        super().__init__(name="A2C", state_dim=state_dim, n_actions=n_actions, seed=seed)
        self.gamma = 0.96
        self.actor_lr = 0.016
        self.critic_lr = 0.024
        self.temperature = 1.0

        self.actor_w = self.rng.normal(0.0, 0.05, size=(self.n_actions, self.state_dim))
        self.actor_b = self.rng.normal(0.0, 0.01, size=(self.n_actions,))
        self.value_w = self.rng.normal(0.0, 0.03, size=(self.state_dim,))
        self.value_b = float(self.rng.normal(0.0, 0.01))

    def _policy(self, state: np.ndarray) -> np.ndarray:
        logits = (self.actor_w @ state + self.actor_b) / self.temperature
        return _softmax(logits)

    def _value(self, state: np.ndarray) -> float:
        return float(self.value_w @ state + self.value_b)

    def act(self, state: np.ndarray, explore: bool) -> ActionDecision:
        probs = self._policy(state)
        if explore:
            action = int(self.rng.choice(self.n_actions, p=probs))
        else:
            action = int(np.argmax(probs))
        return ActionDecision(action=action, action_prob=float(probs[action]))

    def _actor_update(self, state: np.ndarray, action: int, probs: np.ndarray, advantage: float) -> None:
        grad_logits = -probs
        grad_logits[action] += 1.0
        self.actor_w += self.actor_lr * advantage * np.outer(grad_logits, state)
        self.actor_b += self.actor_lr * advantage * grad_logits

    def learn(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        decision: ActionDecision,
    ) -> None:
        del decision
        probs = self._policy(state)
        value = self._value(state)
        next_value = 0.0 if done else self._value(next_state)
        target = float(reward) + self.gamma * next_value
        advantage = target - value

        self.value_w += self.critic_lr * advantage * state
        self.value_b += self.critic_lr * advantage
        self._actor_update(state=state, action=action, probs=probs, advantage=advantage)


class PPOLearner(A2CLearner):
    def __init__(self, state_dim: int, n_actions: int, seed: int) -> None:
        super().__init__(state_dim=state_dim, n_actions=n_actions, seed=seed)
        self.name = "PPO"
        self.clip_ratio = 0.20
        self.actor_lr = 0.014
        self.critic_lr = 0.022

    def learn(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        decision: ActionDecision,
    ) -> None:
        probs = self._policy(state)
        old_prob = max(1e-8, float(decision.action_prob))

        value = self._value(state)
        next_value = 0.0 if done else self._value(next_state)
        target = float(reward) + self.gamma * next_value
        advantage = target - value

        new_prob = max(1e-8, float(probs[action]))
        ratio = new_prob / old_prob

        unclipped = ratio * advantage
        clipped = float(np.clip(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)) * advantage
        if advantage >= 0.0:
            surrogate = min(unclipped, clipped)
        else:
            surrogate = max(unclipped, clipped)

        if abs(advantage) < 1e-10:
            scale = 0.0
        else:
            scale = surrogate / advantage

        self.value_w += self.critic_lr * advantage * state
        self.value_b += self.critic_lr * advantage

        grad_logits = -probs
        grad_logits[action] += 1.0
        self.actor_w += self.actor_lr * advantage * scale * np.outer(grad_logits, state)
        self.actor_b += self.actor_lr * advantage * scale * grad_logits


def build_method(name: str, state_dim: int, n_actions: int, seed: int) -> StrategyLearner:
    key = name.strip().upper()
    if key == "PPO":
        return PPOLearner(state_dim=state_dim, n_actions=n_actions, seed=seed)
    if key == "A2C":
        return A2CLearner(state_dim=state_dim, n_actions=n_actions, seed=seed)
    if key == "DQN":
        return DQNLearner(state_dim=state_dim, n_actions=n_actions, seed=seed)
    raise ValueError(f"Unsupported method '{name}'. Allowed: {', '.join(ALGORITHM_NAMES)}")
