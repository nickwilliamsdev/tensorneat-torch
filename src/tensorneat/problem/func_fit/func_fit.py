import torch

from ..base import BaseProblem
from tensorneat.common import State


class FuncFit(BaseProblem):
    jitable = True

    def __init__(self, error_method: str = "mse"):
        super().__init__()
        if error_method not in {"mse", "rmse", "mae", "mape"}:
            raise ValueError("error_method must be one of mse, rmse, mae, mape")
        self.error_method = error_method
        self._sample_vmap_available = True

    def setup(self, state: State = State()):
        return state

    def evaluate(self, state, randkey, act_func, params):
        del randkey
        predict = self._predict(state, act_func, params)

        if self.error_method == "mse":
            loss = torch.mean((predict - self.targets) ** 2)
        elif self.error_method == "rmse":
            loss = torch.sqrt(torch.mean((predict - self.targets) ** 2))
        elif self.error_method == "mae":
            loss = torch.mean(torch.abs(predict - self.targets))
        elif self.error_method == "mape":
            loss = torch.mean(torch.abs((predict - self.targets) / self.targets))
        else:
            raise NotImplementedError

        return -loss

    def show(self, state, randkey, act_func, params, *args, **kwargs):
        del randkey, args, kwargs
        predict = self._predict(state, act_func, params)
        fitness = self.evaluate(state, None, act_func, params)
        loss = -fitness

        lines = []
        for idx in range(self.inputs.shape[0]):
            lines.append(
                f"input: {self.inputs[idx].tolist()}, target: {self.targets[idx].tolist()}, predict: {predict[idx].tolist()}"
            )
        lines.append(f"loss: {float(loss)}")
        print("\n".join(lines))

    def _predict(self, state, act_func, params):
        if self._sample_vmap_available:
            evaluator = lambda sample: act_func(state, params, sample)
            try:
                return torch.vmap(evaluator)(self.inputs)
            except RuntimeError:
                self._sample_vmap_available = False

        return torch.stack([act_func(state, params, sample) for sample in self.inputs])

    @property
    def inputs(self):
        raise NotImplementedError

    @property
    def targets(self):
        raise NotImplementedError

    @property
    def input_shape(self):
        raise NotImplementedError

    @property
    def output_shape(self):
        raise NotImplementedError