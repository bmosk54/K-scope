"""Claude-on-Bedrock adapter for the agent step (answer -> atomic claims).

Only the *decomposition* step is an LLM call; the causal certification is pure
numpy and never touches the model (the verdict comes from the deterministic
battery). Kept optional and degradable: if boto3 / credentials / model access are
missing, `available()` is False (or `decompose` raises) and callers fall back to
the keyword heuristic in claims.py — so the whole scaffold runs with or without
Bedrock.

Auth is the SageMaker execution-role via SigV4 (no API key). We call the classic
`bedrock-runtime` InvokeModel endpoint (the one the role is entitled to — the newer
Mantle Messages endpoint needs a separate entitlement and 403s here), with the
cross-region inference-profile id. Config via env:
    BEDROCK_MODEL_ID   default us.anthropic.claude-sonnet-4-6
    BEDROCK_REGION     default AWS_REGION, then us-west-2
"""
import json
import os
import re

# Sonnet 4.6 is fast and plenty for claim decomposition (the verdict is the
# battery's job, not the LLM's). On-demand throughput isn't offered for the bare
# id, so we use the cross-region inference profile (`us.` prefix).
DEFAULT_MODEL_ID = "us.anthropic.claude-sonnet-4-6"
ANTHROPIC_VERSION = "bedrock-2023-05-31"


def _region():
    return os.environ.get("BEDROCK_REGION") or os.environ.get("AWS_REGION") or "us-west-2"


def _model_id():
    return os.environ.get("BEDROCK_MODEL_ID") or DEFAULT_MODEL_ID


def _extract_json_array(text):
    """Pull the first JSON array out of a model response (tolerates prose / fences)."""
    fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.S)
    if fenced:
        return json.loads(fenced.group(1))
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end + 1])
    raise ValueError("no JSON array in model response")


class ClaudeBedrock:
    """Thin wrapper over bedrock-runtime InvokeModel. Never raises on construct."""

    def __init__(self, model_id=None, region=None):
        self.model_id = model_id or _model_id()
        self.region = region or _region()
        self._rt = None
        self._err = None
        if os.environ.get("BEDROCK_DISABLE"):
            self._err = "BEDROCK_DISABLE set"
            return
        try:
            import boto3
            self._rt = boto3.client("bedrock-runtime", region_name=self.region)
        except Exception as e:  # boto3 missing / no creds resolvable
            self._err = repr(e)

    def available(self):
        return self._rt is not None

    def _invoke(self, system, user, max_tokens=2000):
        body = {"anthropic_version": ANTHROPIC_VERSION, "max_tokens": max_tokens,
                "system": system,
                "messages": [{"role": "user", "content": user}]}
        r = self._rt.invoke_model(modelId=self.model_id, body=json.dumps(body))
        out = json.loads(r["body"].read())
        return "".join(b.get("text", "") for b in out.get("content", [])
                       if b.get("type") == "text")

    def decompose(self, answer, vocabulary):
        """Split a K-Pro answer into atomic concept-claims.

        Returns a list of {"text", "concept"|None, "polarity"} dicts. Every claim is
        mapped to one concept from `vocabulary` or to null (not certifiable) — the
        caller re-validates against the substrate, so the LLM can't smuggle in a
        concept the substrate can't probe. Raises on any failure so the caller falls
        back to the heuristic.
        """
        if not self.available():
            raise RuntimeError(f"bedrock unavailable: {self._err}")
        vocab = list(vocabulary)
        system = (
            "You decompose a pathology foundation-model answer into ATOMIC, "
            "independently-checkable concept claims. Output ONLY a JSON array. Each "
            "element: {\"text\": <the atomic phrase>, \"concept\": <one name from the "
            "vocabulary, or null if none fits — do NOT force a fit>, \"polarity\": "
            "\"present\" or \"absent\"}. One element per distinct assertion.")
        user = f"Vocabulary: {vocab}\n\nAnswer:\n{answer}\n\nJSON array:"
        text = self._invoke(system, user)
        raw = _extract_json_array(text)
        # normalize + coerce unknown concepts to None
        out = []
        for r in raw:
            c = r.get("concept")
            out.append({"text": str(r.get("text", "")).strip(),
                        "concept": c if c in vocab else None,
                        "polarity": r.get("polarity", "present")})
        return [o for o in out if o["text"]]
