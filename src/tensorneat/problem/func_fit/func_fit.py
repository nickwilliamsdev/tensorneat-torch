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
        self._cpu_dataset = None
        self._dataset_cache = {}

    def setup(self, state: State = State()):
        self._cpu_dataset = (self.inputs, self.targets)
        self._dataset_cache = {}
        return state

    def evaluate(self, state, randkey, act_func, params):
        del randkey
        predict = self._predict(state, act_func, params)
        _, targets = self._prepared_dataset(params)

        if self.error_method == "mse":
            loss = torch.mean((predict - targets) ** 2)
        elif self.error_method == "rmse":
            loss = torch.sqrt(torch.mean((predict - targets) ** 2))
        elif self.error_method == "mae":
            loss = torch.mean(torch.abs(predict - targets))
        elif self.error_method == "mape":
            loss = torch.mean(torch.abs((predict - targets) / targets))
        else:
            raise NotImplementedError

        return -loss

    def show(self, state, randkey, act_func, params, *args, **kwargs):
        del randkey, args, kwargs
        predict = self._predict(state, act_func, params)
        inputs, targets = self._prepared_dataset(params)
        fitness = self.evaluate(state, None, act_func, params)
        loss = -fitness

        lines = []
        for idx in range(inputs.shape[0]):
            lines.append(
                f"input: {inputs[idx].tolist()}, target: {targets[idx].tolist()}, predict: {predict[idx].tolist()}"
            )
        lines.append(f"loss: {float(loss)}")
        print("\n".join(lines))

    def _predict(self, state, act_func, params):
        inputs, _ = self._prepared_dataset(params)
        if self._sample_vmap_available:
            evaluator = lambda sample: act_func(state, params, sample)
            try:
                return torch.vmap(evaluator)(inputs)
            except RuntimeError:
                self._sample_vmap_available = False

        return torch.stack([act_func(state, params, sample) for sample in inputs])

    def _prepared_dataset(self, params):
        if self._cpu_dataset is None:
            self._cpu_dataset = (self.inputs, self.targets)

        cpu_inputs, cpu_targets = self._cpu_dataset
        ref_tensor = self._find_first_tensor(params)
        if ref_tensor is None:
            return cpu_inputs, cpu_targets

        cache_key = (str(ref_tensor.device), ref_tensor.dtype)
        cached = self._dataset_cache.get(cache_key)
        if cached is not None:
            return cached

        prepared = (
            cpu_inputs.to(device=ref_tensor.device, dtype=ref_tensor.dtype),
            cpu_targets.to(device=ref_tensor.device, dtype=ref_tensor.dtype),
        )
        self._dataset_cache[cache_key] = prepared
        return prepared

    def _find_first_tensor(self, value):
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, tuple):
            for item in value:
                found = self._find_first_tensor(item)
                if found is not None:
                    return found
            return None
        if isinstance(value, list):
            for item in value:
                found = self._find_first_tensor(item)
                if found is not None:
                    return found
            return None
        return None

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