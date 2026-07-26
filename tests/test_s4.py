import jax.numpy as jnp

from s4_nnx import (
    S4Block,
    S4Config,
    S4Layer,
    S4LayerEnsemble,
    S4Regressor,
    SequenceBlockNNX,
    StackedModelRegression,
    create_model,
)


def test_public_aliases():
    assert S4Layer is S4LayerEnsemble
    assert S4Block is SequenceBlockNNX
    assert S4Regressor is StackedModelRegression


def test_convolution_forward_shape():
    config = S4Config(
        d_input=3,
        d_output=2,
        d_model=4,
        n_layers=1,
        state_size=8,
        l_max=16,
        decode=False,
    )
    model = create_model(config, seed=0)

    inputs = jnp.ones((16, 3))
    outputs, states = model(inputs, training=False)

    assert outputs.shape == (16, 2)
    assert len(states) == 1
    assert states[0].shape == (4, 8)


def test_recurrent_one_step_shape_and_state():
    config = S4Config(
        d_input=3,
        d_output=2,
        d_model=4,
        n_layers=2,
        state_size=8,
        l_max=16,
        decode=True,
    )
    model = create_model(config, seed=1)
    states = model.init_state()

    one_step_input = jnp.ones(3)
    output, new_states = model(
        one_step_input,
        states=states,
        training=False,
    )

    assert output.shape == (2,)
    assert len(new_states) == 2
    assert all(state.shape == (4, 8) for state in new_states)


def test_config_validation():
    try:
        S4Config(d_input=0, d_output=1)
    except ValueError:
        pass
    else:
        raise AssertionError("S4Config should reject non-positive dimensions")
