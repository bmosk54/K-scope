"""MCP server — exposes the causal evidence-card battery over the Model Context
Protocol. One verb done well: `certify`, plus its sub-verbs.

Run (stdio transport, for Claude Desktop / MCP Inspector):
    python -m biolayer.mcp.server

Register with an MCP client by pointing it at that command. Every tool returns a
JSON-able dict; each causal claim carries its matched-random null.
"""
from mcp.server.fastmcp import FastMCP

from . import verbs

mcp = FastMCP("biolayer-certify")


@mcp.tool()
def certify(model: str = "phikon_v2", split: str = "train",
            pos: str = "TUM", neg: str = "LYM", n_null: int = 200,
            track: str = None) -> dict:
    """Certify a pathology-FM concept prediction: full causal evidence card.

    Pass `track` ("phikon" | "h0") to use that pipeline's model + objective, or
    give (model, pos, neg) directly. Runs every pillar (probe, readout necessity,
    sufficiency/steering, specificity, layer-resolved curve, confound gate)
    against matched-random nulls and returns the structured, auditable evidence
    card with literature grounding + caveat.
    """
    return verbs.certify(model, split, pos, neg, n_null, track=track)


@mcp.tool()
def hypothesis(track: str = "phikon", split: str = "train") -> dict:
    """State the causal hypothesis a track certifies + the ordered verb pipeline."""
    return verbs.hypothesis(track, split)


@mcp.tool()
def certify_answer(prompt: str, answer: str, track: str = "phikon",
                   split: str = "train", n_null: int = 200, fast: bool = False) -> dict:
    """Certify a free-form K-Pro answer, claim by claim (dynamic answer-bound probes).

    Decomposes the answer into atomic concept-claims, resolves each to a labeled
    contrast on the substrate (declining claims with no label as NOT_CERTIFIABLE),
    and runs the full do()-battery + matched-random null + specificity + confound
    gate + held-out validation per claim, with Holm-Bonferroni correction across
    claims. Returns numeric per-pillar scores (necessity/sufficiency/specificity)
    and a GROUNDED/WEAK/NULL verdict per claim. `fast=True` skips the layer sweep.
    """
    return verbs.certify_answer(prompt, answer, track=track, split=split,
                                n_null=n_null, fast=fast)


@mcp.tool()
def probe(model: str = "phikon_v2", split: str = "train",
          pos: str = "TUM", neg: str = "LYM") -> dict:
    """Derive the concept direction and report linear-probe separability."""
    return verbs.probe(model, split, pos, neg)


@mcp.tool()
def attribution(model: str = "phikon_v2", split: str = "train",
                pos: str = "TUM", neg: str = "LYM", mode: str = "soft") -> dict:
    """Patch-level 'hack': rank patches that build the concept global + a new
    concept-focused global embedding, vs a matched-random null."""
    return verbs.attribution_verb(model, split, pos, neg, mode)


@mcp.tool()
def ablate(model: str = "phikon_v2", split: str = "train",
           pos: str = "TUM", neg: str = "LYM", n_null: int = 200) -> dict:
    """Necessity (readout space) with a matched-random null."""
    return verbs.ablate(model, split, pos, neg, n_null)


@mcp.tool()
def specificity(model: str = "phikon_v2", split: str = "train",
                pos: str = "TUM", neg: str = "LYM",
                distractor_pos: str = "STR", distractor_neg: str = "MUS") -> dict:
    """Ablate an orthogonal distractor axis; the target probe should stay intact."""
    return verbs.specificity(model, split, pos, neg, distractor_pos, distractor_neg)


@mcp.tool()
def steer(model: str = "phikon_v2", split: str = "train",
          pos: str = "TUM", neg: str = "LYM", n_null: int = 200) -> dict:
    """Sufficiency: inject the concept direction to flip neg->pos vs a random null."""
    return verbs.steer(model, split, pos, neg, n_null)


@mcp.tool()
def layered(model: str = "phikon_v2", split: str = "train",
            pos: str = "TUM", neg: str = "LYM", space: str = "global") -> dict:
    """Layer-resolved necessity curve across the 3 extracted layers.

    space: "global" (CLS) | "local" (mean patch token). Shows how concept
    separability + readout necessity evolve with depth — the rigor curve.
    """
    return verbs.layered(model, split, pos, neg, space)


@mcp.tool()
def confound(model: str = "phikon_v2", split: str = "train",
             pos: str = "TUM", neg: str = "LYM") -> dict:
    """Confound gate: is the causal axis aligned with a site/scanner signature?

    Returns 'no_multisite_data' until multi-site data lands (track #2).
    """
    return verbs.confound_verb(model, split, pos, neg)


def main():
    mcp.run()  # stdio transport


if __name__ == "__main__":
    main()
