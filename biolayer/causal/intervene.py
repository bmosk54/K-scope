"""Layer-resolved source-intervention necessity — the *rigor* pillar. [IN PROGRESS]

This is the hook-based half of the Bio-Interp Benchmark-D battery that the
readout-space `battery.py` deliberately does NOT cover. Instead of projecting the
concept axis out of the final CLS, it hooks `encoder.layer[L]`, edits the
activations *in place*, and lets the edit propagate forward to the CLS -> probe.

Expected result (RESULTS.md priors, still UNTESTED):
  - mid layers (L=4/10/16): concept-subspace ablation does NOT move the probe
    (Hydra effect / redundant encoding — the model recomputes downstream)
  - near-readout (L=22): ablation finally bites, below the matched-random null
  => the layer-resolved curve is the honest "necessity is redundancy-limited"
     story that separates this from naive single-axis TCAV.

Interface (target — track #3 fills in the body):

    curve = necessity_curve(model_key, pos, neg, layers=(4, 10, 16, 22), n_null=200)
    # -> {"layers": [...], "concept_acc": [...], "random_acc_mean": [...], ...}

Unlike the rest of biolayer.causal this needs the *live encoder* (a forward pass
per intervention), not just the cached .npz — so it imports biolayer.data.models.
"""

# Sentinel so biolayer.mcp.certify can report this pillar's status without crashing.
STATUS = "in_progress"
DEFAULT_LAYERS = (4, 10, 16, 22)


def necessity_curve(model_key, pos="TUM", neg="LYM", layers=DEFAULT_LAYERS,
                    n_null=200, seed=0):
    """Layer-resolved source-intervention necessity curve. NOT YET IMPLEMENTED.

    Track #3: register a forward hook on encoder.layer[L], project the concept
    subspace out of that layer's activations, run the forward pass to the CLS,
    score the frozen probe, and repeat for a matched-random subspace null.
    """
    raise NotImplementedError(
        "necessity_curve: layer-resolved source-intervention not built yet "
        "(track #3). Hook encoder.layer[L], edit activations, propagate to CLS."
    )


def pending_report(layers=DEFAULT_LAYERS):
    """Structured placeholder so the evidence card can show this pillar as pending."""
    return {
        "status": STATUS,
        "layers_planned": list(layers),
        "note": ("layer-resolved necessity curve not yet run; readout-space "
                 "necessity is available in the evidence card's necessity_readout"),
    }
