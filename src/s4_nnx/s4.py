"""Minimal S4 implementation for Flax NNX.

The S4/HiPPO/DPLR mathematics in this module is adapted from the
Annotated S4 implementation. The NNX module structure, explicit recurrent
state handling, and regression wrapper are provided for Flax NNX workflows.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx
from jax.nn.initializers import normal, ones
from jax.numpy.linalg import inv, matrix_power

Array = jax.Array


def scan_ssm(
    a_bar: Array,
    b_bar: Array,
    c_bar: Array,
    inputs: Array,
    initial_state: Array,
) -> tuple[Array, Array]:
    """Run a discrete state-space model with ``jax.lax.scan``.

    Args:
        a_bar: Discrete state matrix with shape ``(N, N)``.
        b_bar: Discrete input matrix with shape ``(N, 1)``.
        c_bar: Discrete output matrix with shape ``(1, N)``.
        inputs: Input sequence with shape ``(L, 1)``.
        initial_state: Initial SSM state with shape ``(N,)``.

    Returns:
        A pair ``(final_state, outputs)``. The outputs have shape ``(L, 1)``.
    """

    def step(previous_state: Array, current_input: Array):
        current_state = a_bar @ previous_state + b_bar @ current_input
        current_output = c_bar @ current_state
        return current_state, current_output

    return jax.lax.scan(step, initial_state, inputs)


def log_step_initializer(
    dt_min: float = 0.001,
    dt_max: float = 0.1,
):
    """Create an initializer that samples uniformly in log-step space."""

    def initialize(key: Array, shape: tuple[int, ...]) -> Array:
        return jax.random.uniform(key, shape) * (
            jnp.log(dt_max) - jnp.log(dt_min)
        ) + jnp.log(dt_min)

    return initialize


def causal_convolution(inputs: Array, kernel: Array) -> Array:
    """Compute one-dimensional causal convolution using FFTs."""
    if inputs.ndim != 1 or kernel.ndim != 1:
        raise ValueError("inputs and kernel must both be rank-1 arrays")
    if kernel.shape[0] != inputs.shape[0]:
        raise ValueError(
            "kernel length must match the input sequence length; "
            f"got {kernel.shape[0]} and {inputs.shape[0]}"
        )

    input_fft = jnp.fft.rfft(jnp.pad(inputs, (0, kernel.shape[0])))
    kernel_fft = jnp.fft.rfft(jnp.pad(kernel, (0, inputs.shape[0])))
    return jnp.fft.irfft(input_fft * kernel_fft)[: inputs.shape[0]]


def constant_initializer(value: Array):
    """Return a JAX initializer that always returns ``value``."""

    def initialize(_key: Array, shape: tuple[int, ...]) -> Array:
        if tuple(shape) != tuple(value.shape):
            raise ValueError(f"expected initializer shape {value.shape}, got {shape}")
        return value

    return initialize


def make_hippo(state_size: int) -> Array:
    """Construct the negative HiPPO matrix."""
    p = jnp.sqrt(1 + 2 * jnp.arange(state_size))
    a = p[:, jnp.newaxis] * p[jnp.newaxis, :]
    a = jnp.tril(a) - jnp.diag(jnp.arange(state_size))
    return -a


def make_nplr_hippo(state_size: int) -> tuple[Array, Array, Array]:
    """Construct the NPLR representation of the HiPPO matrix."""
    a = make_hippo(state_size)
    p = jnp.sqrt(jnp.arange(state_size) + 0.5)
    b = jnp.sqrt(2 * jnp.arange(state_size) + 1.0)
    return a, p, b


def make_dplr_hippo(state_size: int) -> tuple[Array, Array, Array, Array]:
    """Diagonalize the NPLR HiPPO representation."""
    a, p, b = make_nplr_hippo(state_size)
    s = a + p[:, jnp.newaxis] * p[jnp.newaxis, :]

    s_diagonal = jnp.diagonal(s)
    lambda_real = jnp.mean(s_diagonal) * jnp.ones_like(s_diagonal)

    lambda_imag, eigenvectors = jnp.linalg.eigh(s * -1j)
    p = eigenvectors.conj().T @ p
    b = eigenvectors.conj().T @ b

    return lambda_real + 1j * lambda_imag, p, b, eigenvectors


def hippo_initializer(state_size: int):
    """Return fixed initializers for the DPLR HiPPO parameters."""
    lambda_value, p, b, _ = make_dplr_hippo(state_size)
    return (
        constant_initializer(lambda_value.real),
        constant_initializer(lambda_value.imag),
        constant_initializer(p),
        constant_initializer(b),
    )


@jax.jit
def cauchy(v: Array, omega: Array, lambd: Array) -> Array:
    """Evaluate a Cauchy matrix-vector product."""

    def cauchy_dot(single_omega: Array) -> Array:
        return (v / (single_omega - lambd)).sum()

    return jax.vmap(cauchy_dot)(omega)


def kernel_dplr(
    lambd: Array,
    p: Array,
    q: Array,
    b: Array,
    c_vector: Array,
    step: Array,
    sequence_length: int,
) -> Array:
    """Construct the real-valued S4 convolution kernel."""
    omega = jnp.exp(
        (-2j * jnp.pi) * (jnp.arange(sequence_length) / sequence_length)
    )

    a_term = (c_vector.conj(), q.conj())
    b_term = (b, p)

    g = (2.0 / step) * ((1.0 - omega) / (1.0 + omega))
    scale = 2.0 / (1.0 + omega)

    k00 = cauchy(a_term[0] * b_term[0], g, lambd)
    k01 = cauchy(a_term[0] * b_term[1], g, lambd)
    k10 = cauchy(a_term[1] * b_term[0], g, lambd)
    k11 = cauchy(a_term[1] * b_term[1], g, lambd)

    values_at_roots = scale * (k00 - k01 * (1.0 / (1.0 + k11)) * k10)
    return jnp.fft.ifft(values_at_roots, sequence_length).reshape(sequence_length).real


def discrete_dplr(
    lambd: Array,
    p: Array,
    q: Array,
    b: Array,
    c_vector: Array,
    step: Array,
    l_max: int,
) -> tuple[Array, Array, Array]:
    """Convert continuous DPLR parameters to a recurrent discrete SSM."""
    b = b[:, jnp.newaxis]
    c_transpose = c_vector[jnp.newaxis, :]

    state_size = lambd.shape[0]
    identity = jnp.eye(state_size)

    forward_euler = (2.0 / step) * identity + (
        jnp.diag(lambd) - p[:, jnp.newaxis] @ q[:, jnp.newaxis].conj().T
    )

    diagonal_inverse = jnp.diag(1.0 / ((2.0 / step) - lambd))
    q_conjugate = q.conj().T.reshape(1, -1)
    p_column = p.reshape(-1, 1)

    backward_euler = diagonal_inverse - (
        diagonal_inverse
        @ p_column
        * (1.0 / (1.0 + q_conjugate @ diagonal_inverse @ p_column))
        * q_conjugate
        @ diagonal_inverse
    )

    a_bar = backward_euler @ forward_euler
    b_bar = 2 * backward_euler @ b
    c_bar = c_transpose @ inv(identity - matrix_power(a_bar, l_max)).conj()

    return a_bar, b_bar, c_bar.conj()


class S4LayerEnsemble(nnx.Module):
    """Per-channel S4 layer parameters stored in one NNX module.

    ``SequenceBlockNNX`` vmaps over the first parameter axis so that each model
    channel has an independent S4 state-space model.
    """

    def __init__(
        self,
        N: int,
        l_max: int,
        D_MODEL: int,
        decode: bool,
        *,
        rngs: nnx.Rngs,
    ):
        self.N = N
        self.decode = decode
        self.l_max = l_max
        self.D_MODEL = D_MODEL

        init_a_real, init_a_imag, init_p, init_b = hippo_initializer(N)
        init_c = normal(stddev=0.5**0.5)
        init_d = ones
        init_log_step = log_step_initializer()

        vmap_axes = (0, None)
        vmap_init_a_real = jax.vmap(init_a_real, in_axes=vmap_axes)
        vmap_init_a_imag = jax.vmap(init_a_imag, in_axes=vmap_axes)
        vmap_init_p = jax.vmap(init_p, in_axes=vmap_axes)
        vmap_init_b = jax.vmap(init_b, in_axes=vmap_axes)
        vmap_init_c = jax.vmap(init_c, in_axes=vmap_axes)
        vmap_init_d = jax.vmap(init_d, in_axes=vmap_axes)
        vmap_init_log_step = jax.vmap(init_log_step, in_axes=vmap_axes)

        keys = jax.random.split(rngs.params(), 7)
        learning_rate_metadata = {"lr": 0.1}

        self.Lambda_re = nnx.Param(
            vmap_init_a_real(jax.random.split(keys[0], D_MODEL), (N,)),
            metadata=learning_rate_metadata,
        )
        self.Lambda_im = nnx.Param(
            vmap_init_a_imag(jax.random.split(keys[1], D_MODEL), (N,)),
            metadata=learning_rate_metadata,
        )
        self.P = nnx.Param(
            vmap_init_p(jax.random.split(keys[2], D_MODEL), (N,)),
            metadata=learning_rate_metadata,
        )
        self.B = nnx.Param(
            vmap_init_b(jax.random.split(keys[3], D_MODEL), (N,)),
            metadata=learning_rate_metadata,
        )
        self.C_real_imag = nnx.Param(
            vmap_init_c(jax.random.split(keys[4], D_MODEL), (N, 2)),
            metadata=learning_rate_metadata,
        )
        self.D = nnx.Param(
            vmap_init_d(jax.random.split(keys[5], D_MODEL), (1,)),
            metadata=learning_rate_metadata,
        )
        self.log_step = nnx.Param(
            vmap_init_log_step(jax.random.split(keys[6], D_MODEL), (1,)),
            metadata=learning_rate_metadata,
        )

    def __call__(self, inputs: Array, previous_state: Array) -> tuple[Array, Array]:
        """Apply one channel of the S4 layer to a sequence.

        Args:
            inputs: One channel with shape ``(L,)``.
            previous_state: Recurrent state with shape ``(N,)``.

        Returns:
            ``(outputs, new_state)``.
        """
        if inputs.ndim != 1:
            raise ValueError(f"S4 channel input must have shape (L,), got {inputs.shape}")
        if previous_state.ndim != 1:
            raise ValueError(
                f"S4 channel state must have shape (N,), got {previous_state.shape}"
            )

        step = jnp.clip(jnp.exp(self.log_step.value), 0.001, 1.0)
        lambd = (
            jnp.clip(self.Lambda_re.value, None, -1e-4)
            + 1j * self.Lambda_im.value
        )
        c_vector = (
            self.C_real_imag.value[..., 0]
            + 1j * self.C_real_imag.value[..., 1]
        )

        if not self.decode:
            if inputs.shape[0] != self.l_max:
                raise ValueError(
                    "Convolution mode currently requires sequence length to equal "
                    f"l_max={self.l_max}; got {inputs.shape[0]}"
                )

            kernel = kernel_dplr(
                lambd,
                self.P.value,
                self.P.value,
                self.B.value,
                c_vector,
                step,
                self.l_max,
            )
            outputs = causal_convolution(inputs, kernel) + self.D.value * inputs
            return outputs, previous_state

        a_bar, b_bar, c_bar = discrete_dplr(
            lambd,
            self.P.value,
            self.P.value,
            self.B.value,
            c_vector,
            step,
            self.l_max,
        )
        final_state, outputs = scan_ssm(
            a_bar,
            b_bar,
            c_bar,
            inputs[:, jnp.newaxis],
            previous_state,
        )
        outputs = outputs.reshape(-1).real + self.D.value * inputs
        return outputs, final_state


class SequenceBlockNNX(nnx.Module):
    """Residual S4 sequence block implemented with Flax NNX."""

    def __init__(
        self,
        layer_cls: type[nnx.Module],
        layer_args: dict,
        d_model: int,
        dropout: float,
        prenorm: bool = True,
        glu: bool = True,
        decode: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        self.d_model = d_model
        self.prenorm = prenorm
        self.glu = glu
        self.decode = decode
        self.dropout_rate = dropout

        self.seq = layer_cls(
            **layer_args,
            D_MODEL=d_model,
            decode=decode,
            rngs=rngs,
        )

        keys = jax.random.split(rngs.params(), 3)
        self.norm = nnx.LayerNorm(d_model, rngs=nnx.Rngs(params=keys[0]))
        self.out = nnx.Linear(d_model, d_model, rngs=nnx.Rngs(params=keys[1]))
        if self.glu:
            self.out2 = nnx.Linear(
                d_model,
                d_model,
                rngs=nnx.Rngs(params=keys[2]),
            )

        self.drop = nnx.Dropout(dropout, broadcast_dims=[0])

    def __call__(
        self,
        inputs: Array,
        s4_state: Array,
        *,
        rngs: nnx.Rngs | None = None,
        training: bool = False,
    ) -> tuple[Array, Array]:
        if inputs.ndim != 2:
            raise ValueError(
                f"SequenceBlockNNX expects shape (L, d_model), got {inputs.shape}"
            )
        if s4_state.ndim != 2:
            raise ValueError(
                f"S4 state must have shape (d_model, N), got {s4_state.shape}"
            )

        skip = inputs
        x = self.norm(inputs) if self.prenorm else inputs

        sequence_graph, sequence_parameters = nnx.split(self.seq)

        def run_one_channel(parameters, channel_input, channel_state):
            single_channel_layer = nnx.merge(sequence_graph, parameters)
            return single_channel_layer(channel_input, channel_state)

        x, new_s4_state = jax.vmap(
            run_one_channel,
            in_axes=(0, 1, 0),
            out_axes=(1, 0),
        )(sequence_parameters, x, s4_state)

        x = nnx.gelu(x)

        if training and rngs is not None and self.dropout_rate > 0.0:
            x = self.drop(x, rngs=rngs)

        if self.glu:
            gate = jax.nn.sigmoid(self.out2(x))
            x = self.out(x) * gate
        else:
            x = self.out(x)

        if training and rngs is not None and self.dropout_rate > 0.0:
            x = self.drop(x, rngs=rngs)

        x = skip + x
        if not self.prenorm:
            x = self.norm(x)

        return x, new_s4_state


class StackedModelRegression(nnx.Module):
    """Stacked S4 model for sequence-to-sequence regression."""

    def __init__(
        self,
        layer_cls: type[nnx.Module],
        layer_args: dict,
        d_input: int,
        d_output: int,
        d_model: int,
        n_layers: int,
        prenorm: bool = True,
        dropout: float = 0.0,
        decode: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        self.d_input = d_input
        self.d_model = d_model
        self.d_output = d_output
        self.n_layers = n_layers
        self.prenorm = prenorm
        self.decode = decode
        self.dropout = dropout
        self.N = int(layer_args["N"])
        self.l_max = int(layer_args["l_max"])

        keys = jax.random.split(rngs.params(), 3)
        self.encoder = nnx.Linear(
            d_input,
            d_model,
            rngs=nnx.Rngs(params=keys[0]),
        )
        self.decoder = nnx.Linear(
            d_model,
            d_output,
            rngs=nnx.Rngs(params=keys[1]),
        )

        layer_keys = jax.random.split(keys[2], n_layers)
        self.layers = [
            SequenceBlockNNX(
                layer_cls=layer_cls,
                layer_args=layer_args,
                d_model=d_model,
                dropout=dropout,
                prenorm=prenorm,
                decode=decode,
                glu=True,
                rngs=nnx.Rngs(params=layer_keys[index]),
            )
            for index in range(n_layers)
        ]

    def init_state(
        self,
        batch_size: int | None = None,
        *,
        dtype=jnp.complex64,
    ) -> list[Array]:
        """Create zero recurrent states for every S4 block."""
        shape = (self.d_model, self.N)
        if batch_size is not None:
            shape = (batch_size, *shape)
        return [jnp.zeros(shape, dtype=dtype) for _ in range(self.n_layers)]

    def __call__(
        self,
        inputs: Array,
        states: list[Array] | None = None,
        *,
        rngs: nnx.Rngs | None = None,
        training: bool = False,
    ) -> tuple[Array, list[Array]]:
        """Run an unbatched sequence through the model.

        Args:
            inputs: Array with shape ``(L, d_input)``. A rank-1 input is treated
                as a one-step sequence.
            states: One recurrent state per S4 block. If omitted, zero states
                are created automatically.
            rngs: Optional dropout RNG stream.
            training: Enables dropout when true.
        """
        was_rank_one = inputs.ndim == 1
        if was_rank_one:
            inputs = inputs[jnp.newaxis, :]
        elif inputs.ndim != 2:
            raise ValueError(
                "StackedModelRegression expects an unbatched input with shape "
                f"(L, d_input); got {inputs.shape}"
            )

        if inputs.shape[-1] != self.d_input:
            raise ValueError(
                f"expected d_input={self.d_input}, got {inputs.shape[-1]}"
            )

        current_states = self.init_state() if states is None else states
        if len(current_states) != self.n_layers:
            raise ValueError(
                f"expected {self.n_layers} state arrays, got {len(current_states)}"
            )

        x = self.encoder(inputs)
        new_states: list[Array] = []

        for layer, state in zip(self.layers, current_states):
            x, new_state = layer(
                x,
                state,
                rngs=rngs,
                training=training,
            )
            new_states.append(new_state)

        output = self.decoder(x)
        if was_rank_one:
            output = output.squeeze(0)

        return output, new_states


@dataclass(frozen=True, slots=True)
class S4Config:
    """Configuration for ``create_model``."""

    d_input: int
    d_output: int
    d_model: int = 16
    n_layers: int = 1
    state_size: int = 32
    l_max: int = 100
    dropout: float = 0.0
    prenorm: bool = True
    decode: bool = False

    def __post_init__(self):
        integer_fields = {
            "d_input": self.d_input,
            "d_output": self.d_output,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "state_size": self.state_size,
            "l_max": self.l_max,
        }
        for name, value in integer_fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the interval [0, 1)")


def create_model(config: S4Config, *, seed: int = 0) -> StackedModelRegression:
    """Create an S4 regression model from a compact configuration."""
    return StackedModelRegression(
        layer_cls=S4LayerEnsemble,
        layer_args={"N": config.state_size, "l_max": config.l_max},
        d_input=config.d_input,
        d_output=config.d_output,
        d_model=config.d_model,
        n_layers=config.n_layers,
        prenorm=config.prenorm,
        dropout=config.dropout,
        decode=config.decode,
        rngs=nnx.Rngs(
            params=jax.random.PRNGKey(seed),
            dropout=jax.random.PRNGKey(seed + 1),
        ),
    )


# Short public names while retaining compatibility with the research code.
S4Layer = S4LayerEnsemble
S4Block = SequenceBlockNNX
S4Regressor = StackedModelRegression

# Backward-compatible helper names used in the original research notebook.
scan_SSM = scan_ssm
make_HiPPO = make_hippo
make_NPLR_HiPPO = make_nplr_hippo
make_DPLR_HiPPO = make_dplr_hippo
kernel_DPLR = kernel_dplr
discrete_DPLR = discrete_dplr
