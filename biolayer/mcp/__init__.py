"""biolayer.mcp — the MCP surface: certify(prediction) -> causal evidence card.

One verb done well, per STRATEGY.md: `certify`, decomposed into sub-verbs
`probe`, `ablate`, `specificity`, `steer`, and `confound`. Each wraps the frozen
biolayer.causal battery and emits a slice of the structured evidence card; every
claim carries its matched-random null. Run the server with:

    python -m biolayer.mcp.server
"""

__all__ = ["verbs"]
