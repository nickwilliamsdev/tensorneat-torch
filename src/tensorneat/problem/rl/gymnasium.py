import torch

from .rl_env import RLEnv


class GymnasiumEnv(RLEnv):
    def __init__(self, env_name: str, env_kwargs=None, clip_actions: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            import gymnasium as gym
        except ImportError as exc:
            raise ImportError("GymnasiumEnv requires gymnasium to be installed") from exc

        self.gym = gym
        self.env_name = env_name
        self.env_kwargs = env_kwargs or {}
        self.clip_actions = clip_actions
        self.env = gym.make(env_name, **self.env_kwargs)

        observation_shape = getattr(self.env.observation_space, "shape", None)
        if observation_shape is None:
            raise TypeError("GymnasiumEnv currently supports observation spaces with a shape attribute")
        self._input_shape = tuple(observation_shape)

        action_space = self.env.action_space
        if getattr(action_space, "shape", None):
            self._output_shape = tuple(action_space.shape)
            self._action_kind = "box"
            self._action_low = torch.as_tensor(action_space.low, dtype=torch.float32)
            self._action_high = torch.as_tensor(action_space.high, dtype=torch.float32)
        elif hasattr(action_space, "n"):
            self._output_shape = (int(action_space.n),)
            self._action_kind = "discrete"
            self._action_low = None
            self._action_high = None
        else:
            raise TypeError("GymnasiumEnv currently supports Box and Discrete action spaces")

    def env_reset(self, seed):
        obs, _ = self.env.reset(seed=int(seed))
        return torch.as_tensor(obs, dtype=torch.float32), None

    def env_step(self, seed, env_state, action):
        del seed, env_state
        obs, reward, terminated, truncated, info = self.env.step(self._format_action(action))
        return (
            torch.as_tensor(obs, dtype=torch.float32),
            None,
            torch.tensor(float(reward), dtype=torch.float32),
            bool(terminated or truncated),
            info,
        )

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def output_shape(self):
        return self._output_shape

    def _format_action(self, action):
        action_tensor = torch.as_tensor(action, dtype=torch.float32).detach().cpu()
        if self._action_kind == "box":
            action_tensor = action_tensor.reshape(self._output_shape)
            if self.clip_actions:
                action_tensor = torch.clamp(action_tensor, self._action_low, self._action_high)
            return action_tensor.numpy()

        if action_tensor.numel() == 1:
            return int(torch.clamp(torch.round(action_tensor.reshape(())), 0, self._output_shape[0] - 1).item())
        return int(torch.argmax(action_tensor).item())