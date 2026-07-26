# S4 for Flax NNX

A minimal implementation of the Structured State Space Sequence model (S4)
using the Flax NNX API.

This repository does **not** introduce a new S4 algorithm. Its purpose is to
provide an NNX-compatible implementation with:

- `flax.nnx.Module` and `flax.nnx.Param`
- explicit recurrent-state input and output
- convolutional training mode and recurrent inference mode
- JAX transformations such as `jit`, `vmap`, and `lax.scan`
- a small sequence-to-sequence regression wrapper

## Installation

From the repository root:

```bash
pip install -e .
```

Install the optional training dependencies:

```bash
pip install -e ".[train]"
```

After pushing the repository to GitHub, it can also be installed with:

```bash
pip install "s4-nnx @ git+https://github.com/YOUR_GITHUB_USERNAME/s4-nnx.git"
```

Replace `YOUR_GITHUB_USERNAME` in this README and in `pyproject.toml`.

## Basic usage

```python
import jax.numpy as jnp

from s4_nnx import S4Config, create_model

config = S4Config(
    d_input=9,
    d_output=6,
    d_model=16,
    n_layers=1,
    state_size=32,
    l_max=100,
    decode=False,
)

model = create_model(config, seed=0)

inputs = jnp.ones((100, 9))
outputs, states = model(inputs, training=False)

print(outputs.shape)       # (100, 6)
print(states[0].shape)     # (16, 32)
```

Convolution mode currently expects the sequence length to equal `l_max`.

## Recurrent inference

Set `decode=True` and pass the returned states into the next call:

```python
import jax.numpy as jnp

from s4_nnx import S4Config, create_model

config = S4Config(
    d_input=9,
    d_output=6,
    d_model=16,
    n_layers=1,
    state_size=32,
    l_max=100,
    decode=True,
)

model = create_model(config, seed=0)
states = model.init_state()

z_k = jnp.ones(9)
x_next, states = model(z_k, states=states, training=False)
```

The recurrent state is passed explicitly. The model does not mutate a hidden
state stored inside the module.

## Existing research-code names

The shorter public names and your existing class names are both available:

| Short name | Existing name |
|---|---|
| `S4Layer` | `S4LayerEnsemble` |
| `S4Block` | `SequenceBlockNNX` |
| `S4Regressor` | `StackedModelRegression` |

## Batched training

The model's direct call accepts one sequence with shape `(L, d_input)`.
Use `nnx.vmap` to run a shared model over a batch. See
[`examples/regression.py`](examples/regression.py).

## Tested environment

The implementation was run on Kaggle with:

| Library | Version |
|---|---:|
| JAX | 0.7.2 |
| Flax | 0.11.2 |
| Optax | 0.2.8 |
| tqdm | 4.67.3 |
| Matplotlib | 3.10.0 |
| NumPy | 2.4.6 |
| SciPy | 1.16.3 |

The exact environment is recorded in `requirements-kaggle.txt`. The core
package itself only imports JAX and Flax.

## Run the tests

```bash
pip install -e ".[dev]"
pytest
```

## Acknowledgment

The S4/HiPPO/DPLR mathematics is adapted from the MIT-licensed
`srush/annotated-s4` implementation. The main purpose of this repository is
the Flax NNX port and explicit recurrent-state interface.

## Status

This is an early research release. The API and checkpoint format may change
before version 1.0.

## License

MIT. See `LICENSE`.
