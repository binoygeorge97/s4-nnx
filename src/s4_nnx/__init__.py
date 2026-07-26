"""Public API for s4-nnx."""

from .s4 import (
    S4Block,
    S4Config,
    S4Layer,
    S4LayerEnsemble,
    S4Regressor,
    SequenceBlockNNX,
    StackedModelRegression,
    causal_convolution,
    create_model,
    discrete_dplr,
    kernel_dplr,
    make_dplr_hippo,
    make_hippo,
    make_nplr_hippo,
    scan_ssm,
)

__all__ = [
    "S4Block",
    "S4Config",
    "S4Layer",
    "S4LayerEnsemble",
    "S4Regressor",
    "SequenceBlockNNX",
    "StackedModelRegression",
    "causal_convolution",
    "create_model",
    "discrete_dplr",
    "kernel_dplr",
    "make_dplr_hippo",
    "make_hippo",
    "make_nplr_hippo",
    "scan_ssm",
]

__version__ = "0.1.0"
