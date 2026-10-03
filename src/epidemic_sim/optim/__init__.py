"""Framework-independent algorithmic kernels (numpy/scipy only).

They are NOT used by the Mesa model: the Mesa implementation relies on Mesa's own tools.
They exist to measure the algorithm separately from the framework (same kernel can be
plugged in the ECS and in Mesa), see ``experiments/optim``.
"""
