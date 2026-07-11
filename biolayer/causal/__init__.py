"""biolayer.causal — the frozen Bio-Interp causal battery, ported to pathology FMs.

Pillars, each certified against a matched-random null (the Section-5-D control):

    probe        derive the concept direction on frozen CLS features
    battery      readout-space necessity + sufficiency(steering) + specificity
    intervene    layer-resolved source-intervention necessity curve   [in progress]
    confound     Kömen-style site/scanner probe on the causal axis     [needs multi-site]

`battery.run_battery` is the load-bearing, working entry point today. `intervene`
and `confound` are the two open pillars the MCP `certify` verb reports as pending
until their tracks land.
"""

from . import probe  # noqa: F401
from .battery import run_battery  # noqa: F401

__all__ = ["probe", "run_battery"]
