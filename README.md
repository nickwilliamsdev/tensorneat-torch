# pytorch-tensorneat

Incremental PyTorch port of the original JAX-based `tensorneat` package.

Current status:

- Common, genome, NEAT, pipeline, function-fit problems, and HyperNEAT feedforward paths are ported
- A basic RL problem surface is ported with a generic rollout base, a packaged mock environment, and an optional Gymnasium adapter
- Project-local `.venv` is set up and validated on this machine
- Example scripts are available under `examples/func_fit` and `examples/rl`
- Unittests cover NEAT, genome operators, RL rollout integration, and HyperNEAT behavior

Quick start:

```bash
python -m unittest discover -s tests -p 'test_*.py'
python examples/func_fit/xor.py
python examples/func_fit/xor_hyperneat_feedforward.py
python examples/rl/mock_target_tracking.py
```

Remaining work is mostly deeper coverage and parity work, such as more problem implementations, richer visualization/sympy features, and broader example/test coverage.