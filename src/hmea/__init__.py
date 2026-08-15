"""Tools for studying adaptive update multipliers in tabular Q-learning."""

from .agent import QAgent, TrainLog
from .envs import ChainMDP
from .homeostasis import HomeostaticState
from .modulation import (
    ClippedModulation,
    HomeostaticModulation,
    LaggedModulation,
    Modulator,
    NoModulation,
    OnlineModulation,
)
from .validation import (
    ModulatorAudit,
    audit,
    check_bounded,
    check_current_error_independence,
    validate_modulator,
)

__version__ = "0.1.0"

__all__ = [
    "ChainMDP",
    "ClippedModulation",
    "HomeostaticModulation",
    "HomeostaticState",
    "LaggedModulation",
    "Modulator",
    "ModulatorAudit",
    "NoModulation",
    "OnlineModulation",
    "QAgent",
    "TrainLog",
    "__version__",
    "audit",
    "check_bounded",
    "check_current_error_independence",
    "validate_modulator",
]
