# S4 for Flax NNX

A minimal implementation of the Structured State Space Sequence model (S4) using the Flax NNX API.

This package focuses on providing an NNX-compatible S4 implementation rather than introducing a new S4 architecture. It uses:

* `flax.nnx.Module` and `flax.nnx.Param`
* explicit recurrent-state input and output
* convolutional and recurrent execution modes
* JAX transformations such as `jit`, `vmap`, and `lax.scan`
* a small stacked regression model for sequence-to-sequence prediction

## Installation

Install directly from GitHub:

```bash
pip install "s4-nnx @ git+https://github.com/YOUR_USERNAME/s4-nnx.git"
```

For development:

```bash
git clone https://github.com/YOUR_USERNAME/s4-nnx.git
cd s4-nnx
pip install -e .
```

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

x = jnp.ones((100, 9))
states = model.init_state()

y, new_states = model(
    x,
    states=states,
    training=False,
)

print(y.shape)
```

## Recurrent inference

Set `decode=True` to use the recurrent S4 realization:

```python
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

z_k = jnp.ones((1, 9))

x_next, states = model(
    z_k,
    states=states,
    training=False,
)
```

The recurrent state is passed explicitly rather than stored and mutated inside the module.

## Tested environment

The current implementation has been tested on Kaggle with:

| Library    | Version |
| ---------- | ------: |
| JAX        |   0.7.2 |
| Flax       |  0.11.2 |
| Optax      |   0.2.8 |
| NumPy      |   2.4.6 |
| SciPy      |  1.16.3 |
| Matplotlib |  3.10.0 |
| tqdm       |  4.67.3 |

## Scope

The package contains the reusable S4 model implementation.

Dataset generation, system-identification experiments, controller training, and application-specific code are intentionally kept outside the core package.

## Status

This is an early research release. The public API and checkpoint format may change before version 1.0.

## License

MIT
