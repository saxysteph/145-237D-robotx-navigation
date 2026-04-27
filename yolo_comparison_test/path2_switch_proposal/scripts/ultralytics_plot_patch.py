"""Limit Ultralytics batch mosaic plots to fewer subplots so each tile stays readable."""

from __future__ import annotations

import functools
from typing import Any

_wrapped_ref: Any = None


def apply_plot_max_subplots(max_subplots: int = 4) -> None:
    """Patch ultralytics.utils.plotting.plot_images default max_subplots (default 16). Safe to call once."""
    global _wrapped_ref
    from ultralytics.utils import plotting as ul_plot

    if _wrapped_ref is not None:
        return

    _orig = ul_plot.plot_images

    @functools.wraps(_orig)
    def _plot_images_limited(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("max_subplots", max_subplots)
        return _orig(*args, **kwargs)

    ul_plot.plot_images = _plot_images_limited
    _wrapped_ref = _plot_images_limited
