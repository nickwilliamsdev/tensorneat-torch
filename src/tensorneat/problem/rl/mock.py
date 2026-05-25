import torch

from .rl_env import RLEnv


class TargetTrackingEnv(RLEnv):
    def __init__(self, target_scale: float = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_scale = float(target_scale)

    def env_reset(self, seed):
        del seed
        return torch.tensor([0.0], dtype=torch.float32), {"step": 0}

    def env_step(self, seed, env_state, action):
        del seed
        step = env_state["step"] + 1
        target = self.target_scale * step / self.max_step
        obs = torch.tensor([target], dtype=torch.float32)
        action_value = torch.as_tensor(action, dtype=torch.float32).reshape(-1)[0]
        reward = 1.0 - torch.abs(action_value - target)
        done = step >= self.max_step
        return obs, {"step": step}, reward, done, {"target": target}

    @property
    def input_shape(self):
        return (1,)

    @property
    def output_shape(self):
        return (1,)