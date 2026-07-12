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
                   split: str = "train", n_null: int = 200, fast: bool = False,
                   explain: bool = False) -> dict:
    """Certify a free-form K-Pro answer, claim by claim (dynamic answer-bound probes).

    Decomposes the answer into atomic concept-claims, resolves each to a labeled
    contrast on the substrate (declining claims with no label as NOT_CERTIFIABLE),
    and runs the full do()-battery + matched-random null + specificity + confound
    gate + held-out validation per claim, with Holm-Bonferroni correction across
    claims. Returns numeric per-pillar scores (necessity/sufficiency/specificity)
    and a GROUNDED/WEAK/NULL verdict per claim, each with a deterministic per-claim
    reasoning_trace (numbers + why). `fast=True` skips the layer sweep; `explain=True`
    adds a plain-English narration in one extra batched LLM call.
    """
    return verbs.certify_answer(prompt, answer, track=track, split=split,
                                n_null=n_null, fast=fast, explain=explain)


@mcp.tool()
def design(question: str, model: str = "phikon_v2", split: str = "train",
           track: str = None, max_probes: int = 8, certify_each: bool = False,
           n_null: int = 200) -> dict:
    """Let Sonnet DESIGN a causal probe battery for a free-form pathology question, then
    optionally certify each proposed probe (generate-then-certify).

    The LLM only proposes (pos, neg, distractor) contrasts over the substrate's real tissue
    classes; the deterministic battery decides certifiability, so an ill-posed probe is
    never certified. This is the generative counterpart to the static `hypothesis` verb.
    Set `certify_each=True` to run the full evidence card for every proposed probe.
    """
    return verbs.design(question, model, split, track=track, max_probes=max_probes,
                        certify_each=certify_each, n_null=n_null)


@mcp.tool()
def rehypothesize(model: str = "phikon_v2", split: str = "train",
                  pos: str = "TUM", neg: str = "LYM", n_null: int = 200,
                  track: str = None) -> dict:
    """Close the certify loop: certify a concept, read its SCORE + REASONING TRACE, and have
    Sonnet propose the NEXT hypothesis + a concrete follow-up probe + a message to feed
    downstream to K-Pro or another Claude.

    Turns a single evidence card into the next iteration: certify -> (score + trace) ->
    Claude -> next_hypothesis -> feed to K-Pro/Claude -> certify. The proposed probe is
    validated against real classes; the battery still decides certifiability.
    """
    return verbs.rehypothesize(model, split, pos, neg, n_null=n_null, track=track)


@mcp.tool()
def steer_from_card(model: str = "phikon_v2", split: str = "train",
                    pos: str = "TUM", neg: str = "LYM", alpha: float = None,
                    track: str = None, x: list = None) -> dict:
    """Zero-recompute sufficiency: steer features toward `pos` using ONLY the direction
    handles a prior `certify` persisted (no probe refit — the card's reuse path).

    Pass raw CLS feature rows in `x`, or omit `x` to steer the split's `neg`-class rows as a
    self-contained demo and report the flip-to-pos rate. Run `certify` first to persist the
    handles.
    """
    return verbs.steer_from_card(x=x, model=model, split=split, pos=pos, neg=neg,
                                 alpha=alpha, track=track)


@mcp.tool()
def ablate_from_card(model: str = "phikon_v2", split: str = "train",
                     pos: str = "TUM", neg: str = "LYM", track: str = None,
                     x: list = None) -> dict:
    """Zero-recompute necessity: project the concept axis out using ONLY the direction
    handles a prior `certify` persisted (no probe refit — the card's reuse path).

    Pass raw CLS feature rows in `x`, or omit `x` to ablate the split's `pos`-class rows and
    report how many still read as `pos` (necessity: this should drop). Run `certify` first.
    """
    return verbs.ablate_from_card(x=x, model=model, split=split, pos=pos, neg=neg,
                                  track=track)


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


@mcp.tool()
def embed(images: list = None, s3_tiles: list = None, slide_s3: str = None,
          keys: list = None, push_index: str = None, slide_name: str = "query",
          max_tiles: int = 16, mpp: float = 0.5) -> dict:
    """Embed NEW pathology tiles on demand through the warm H-optimus-0 endpoint.

    The live bridge into the frozen substrate: turn fresh tissue — base64 tile bytes
    (`images`), S3 tile keys (`s3_tiles`), or a slide URI (`slide_s3`, bounded by
    `max_tiles`) — into 1536-d CLS vectors WITHOUT re-downloading the 4 GB model per call
    (it stays warm on a g5 endpoint). Set `push_index` to also write the vectors into that
    h0-vector index so a query tile becomes searchable alongside the cohort. Provide exactly
    one source. Returns status='unavailable' (never errors) if the endpoint isn't deployed.
    """
    return verbs.embed(images=images, s3_tiles=s3_tiles, slide_s3=slide_s3, keys=keys,
                       push_index=push_index, slide_name=slide_name,
                       max_tiles=max_tiles, mpp=mpp)


def main():
    mcp.run()  # stdio transport


if __name__ == "__main__":
    main()
