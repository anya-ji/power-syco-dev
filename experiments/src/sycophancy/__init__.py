"""Status-based sycophancy experiments on SAGE-Eval prompts.

Pipeline: ``dataset`` -> ``statuses`` -> ``generation`` (vLLM)
-> ``judge`` (SAGE rubric) -> ``analysis`` -> ``plots`` / ``report``.
"""

from .config import CONDITIONS, MODEL_VARIANTS, ExperimentConfig, SamplingConfig

__all__ = ["CONDITIONS", "MODEL_VARIANTS", "ExperimentConfig", "SamplingConfig"]
__version__ = "0.3.0"
