from typing import Callable

import torch

from tensorneat.common import State

from ..base import BaseProblem


class RLEnv(BaseProblem):
    jitable = False

    def __init__(
        self,
        max_step: int = 1000,
        repeat_times: int = 1,
        action_policy: Callable | None = None,
        obs_normalization: bool = False,
        sample_policy: Callable | None = None,
        sample_episodes: int = 0,
    ):
        super().__init__()
        self.max_step = max_step
        self.repeat_times = repeat_times
        self.action_policy = action_policy
        self.obs_normalization = obs_normalization
        self.sample_policy = sample_policy
        self.sample_episodes = sample_episodes

        if self.obs_normalization:
            if self.sample_policy is None:
                raise ValueError("sample_policy must be provided when obs_normalization is enabled")
            if self.sample_episodes <= 0:
                raise ValueError("sample_episodes must be greater than 0 when obs_normalization is enabled")

    def setup(self, state: State = State()):
        if not self.obs_normalization:
            return state

        observations = []
        for episode_idx in range(self.sample_episodes):
            _, episode = self._evaluate_once(
                state,
                self._seed_to_int(getattr(state, "randkey", 0)) + episode_idx,
                lambda current_state, params, obs: obs,
                None,
                lambda seed, forward_func, obs: self.sample_policy(seed, obs),
                record_episode=True,
                normalize_obs=False,
            )
            if episode["obs"].numel() > 0:
                observations.append(episode["obs"])

        if not observations:
            raise ValueError("Failed to collect observations for normalization")

        obs_tensor = torch.cat(observations, dim=0)
        obs_mean = obs_tensor.mean(dim=0)
        obs_std = obs_tensor.std(dim=0, unbiased=False)
        return state.register(problem_obs_mean=obs_mean, problem_obs_std=obs_std)

    def evaluate(self, state: State, randkey, act_func: Callable, params):
        base_seed = self._seed_to_int(randkey)
        rewards = [
            self._evaluate_once(
                state,
                base_seed + repeat_idx,
                act_func,
                params,
                self.action_policy,
                record_episode=False,
                normalize_obs=self.obs_normalization,
            )
            for repeat_idx in range(self.repeat_times)
        ]
        rewards = torch.stack([torch.as_tensor(reward, dtype=torch.float32) for reward in rewards])
        return rewards.mean()

    def _evaluate_once(
        self,
        state: State,
        seed: int,
        act_func: Callable,
        params,
        action_policy: Callable | None,
        record_episode: bool,
        normalize_obs: bool,
    ):
        obs, env_state = self.reset(seed)
        obs = self._to_tensor(obs)

        total_reward = torch.tensor(0.0, dtype=torch.float32)
        obs_history = []
        action_history = []
        reward_history = []

        done = False
        count = 0
        while not done and count < self.max_step:
            current_obs = normalize_observation(state, obs) if normalize_obs else obs
            if action_policy is None:
                action = act_func(state, params, current_obs)
            else:
                action = action_policy(
                    seed + count,
                    lambda current: act_func(state, params, current),
                    current_obs,
                )
            action = self._to_tensor(action)

            next_obs, env_state, reward, done, _ = self.step(seed + count + 1, env_state, action)
            next_obs = self._to_tensor(next_obs)
            reward = torch.as_tensor(reward, dtype=torch.float32)

            if record_episode:
                obs_history.append(current_obs.clone())
                action_history.append(action.clone())
                reward_history.append(reward.clone())

            total_reward = total_reward + reward
            obs = next_obs
            done = bool(done)
            count += 1

        if not record_episode:
            return total_reward

        return total_reward, {
            "obs": self._stack_or_empty(obs_history, self.input_shape),
            "action": self._stack_or_empty(action_history, self.output_shape),
            "reward": self._stack_or_empty(reward_history, ()),
        }

    def step(self, seed, env_state, action):
        return self.env_step(seed, env_state, action)

    def reset(self, seed):
        return self.env_reset(seed)

    def env_step(self, seed, env_state, action):
        raise NotImplementedError

    def env_reset(self, seed):
        raise NotImplementedError

    @property
    def input_shape(self):
        raise NotImplementedError

    @property
    def output_shape(self):
        raise NotImplementedError

    def show(self, state, randkey, act_func, params, *args, **kwargs):
        del args, kwargs
        total_reward, episode = self._evaluate_once(
            state,
            self._seed_to_int(randkey),
            act_func,
            params,
            self.action_policy,
            record_episode=True,
            normalize_obs=self.obs_normalization,
        )
        print(f"total reward: {float(total_reward):.6f}")
        for idx in range(episode["reward"].shape[0]):
            print(
                f"step: {idx}, obs: {episode['obs'][idx].tolist()}, action: {episode['action'][idx].tolist()}, reward: {float(episode['reward'][idx])}"
            )

    def _stack_or_empty(self, values, tail_shape):
        if not values:
            return torch.empty((0, *tail_shape), dtype=torch.float32)
        return torch.stack([torch.as_tensor(value, dtype=torch.float32) for value in values])

    def _seed_to_int(self, seed) -> int:
        if isinstance(seed, torch.Tensor):
            return int(seed.item())
        return int(seed)

    def _to_tensor(self, value):
        return torch.as_tensor(value, dtype=torch.float32)


def normalize_observation(state: State, obs: torch.Tensor):
    if "problem_obs_mean" not in state or "problem_obs_std" not in state:
        return obs
    return (obs - state.problem_obs_mean) / (state.problem_obs_std + 1e-6)