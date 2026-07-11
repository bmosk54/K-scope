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
            pos: str = "TUM", neg: str = "LYM", n_null: int = 200) -> dict:
    """Certify a pathology-FM concept prediction: full causal evidence card.

    Runs every pillar (probe, readout necessity, sufficiency/steering,
    specificity, confound gate) against matched-random nulls and returns the
    structured, auditable evidence card with literature grounding + caveat.
    """
    return verbs.certify(model, split, pos, neg, n_null)


@mcp.tool()
def probe(model: str = "phikon_v2", split: str = "train",
          pos: str = "TUM", neg: str = "LYM") -> dict:
    """Derive the concept direction and report linear-probe separability."""
    return verbs.probe(model, split, pos, neg)


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
