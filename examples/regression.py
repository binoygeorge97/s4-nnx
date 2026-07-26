"""Minimal batched regression example for s4-nnx.

Run from the repository root after installing the training extra:

    pip install -e ".[train]"
    python examples/regression.py
"""

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from s4_nnx import S4Config, create_model


batched_forward = nnx.vmap(
    lambda model, inputs, dropout_key, is_training: model(
        inputs,
        states=None,
        rngs=nnx.Rngs(dropout=dropout_key),
        training=is_training,
    ),
    in_axes=(nnx.StateAxes({nnx.Param: None}), 0, 0, None),
    out_axes=0,
)


def make_linear_system_dataset(
    key: jax.Array,
    *,
    batch_size: int,
    sequence_length: int,
) -> tuple[jax.Array, jax.Array]:
    """Generate sequences from a stable two-state linear system."""
    a = jnp.array([[0.95, 0.05], [-0.02, 0.90]])
    b = jnp.array([[0.10], [0.05]])

    state_key, input_key = jax.random.split(key)
    initial_states = jax.random.uniform(
        state_key,
        (batch_size, 2),
        minval=-1.0,
        maxval=1.0,
    )
    control_sequences = jax.random.uniform(
        input_key,
        (batch_size, sequence_length, 1),
        minval=-1.0,
        maxval=1.0,
    )

    def rollout(initial_state, controls):
        def step(state, control):
            next_state = a @ state + b @ control
            model_input = jnp.concatenate([state, control], axis=-1)
            return next_state, (model_input, next_state)

        _, (inputs, targets) = jax.lax.scan(step, initial_state, controls)
        return inputs, targets

    return jax.vmap(rollout)(initial_states, control_sequences)


@nnx.jit
def train_step(model, optimizer, inputs, targets, dropout_keys):
    def loss_fn(current_model):
        predictions, _ = batched_forward(
            current_model,
            inputs,
            dropout_keys,
            True,
        )
        return jnp.mean((predictions - targets) ** 2)

    loss, gradients = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, gradients)
    return loss


def main():
    seed = 0
    sequence_length = 32
    batch_size = 64

    data_key, training_key = jax.random.split(jax.random.PRNGKey(seed))
    inputs, targets = make_linear_system_dataset(
        data_key,
        batch_size=batch_size,
        sequence_length=sequence_length,
    )

    config = S4Config(
        d_input=3,
        d_output=2,
        d_model=8,
        n_layers=1,
        state_size=16,
        l_max=sequence_length,
        dropout=0.0,
        decode=False,
    )
    model = create_model(config, seed=seed)
    optimizer = nnx.Optimizer(
        model,
        optax.adam(learning_rate=1e-3),
        wrt=nnx.Param,
    )

    for epoch in range(101):
        training_key, dropout_key = jax.random.split(training_key)
        dropout_keys = jax.random.split(dropout_key, batch_size)
        loss = train_step(
            model,
            optimizer,
            inputs,
            targets,
            dropout_keys,
        )

        if epoch % 20 == 0:
            print(f"epoch={epoch:03d} mse={float(loss):.6e}")


if __name__ == "__main__":
    main()

