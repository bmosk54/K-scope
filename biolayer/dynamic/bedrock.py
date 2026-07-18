"""LLM adapter for the agent step (answer -> atomic claims).

Only the *decomposition* / *design* / *reflection* steps are LLM calls; the causal
certification is pure numpy and never touches the model (the verdict comes from the
deterministic battery). Kept optional and degradable: if the provider is unreachable
(no key, no boto3 creds, model access missing), `available()` is False (or the call
raises) and callers fall back to the keyword heuristic in claims.py / probe_design.py
— so the whole scaffold runs with or without a live LLM.

Two interchangeable providers, selected by `LLM_PROVIDER` (env, default "openai"):

  - "openai"   -> the OpenAI API with your own API key (OPENAI_API_KEY). This is the
                  default for a local/laptop deployment — no AWS account needed.
                  NOTE: a ChatGPT Plus/Pro *subscription* does NOT include API access.
                  API calls are billed separately via an API key from
                  https://platform.openai.com/api-keys (with its own billing set up
                  at https://platform.openai.com/account/billing).
  - "bedrock"  -> Claude on AWS Bedrock via the SageMaker execution-role (SigV4, no
                  API key) — the original hackathon path, kept for the team's shared
                  AWS setup. Calls the classic `bedrock-runtime` InvokeModel endpoint
                  with the cross-region inference-profile id.

Config via env:
    LLM_PROVIDER       "openai" (default) | "bedrock"
    OPENAI_API_KEY     required for the openai provider
    OPENAI_MODEL        default gpt-5.6-terra
    BEDROCK_MODEL_ID   default us.anthropic.claude-sonnet-4-6
    BEDROCK_REGION     default AWS_REGION, then us-west-2

Every caller in this codebase constructs `ClaudeBedrock()` (kept as the class name
for zero call-site churn) and only uses `.available()` / `._invoke()` / the
higher-level `.decompose()` / `.narrate()` / `.propose_hypothesis()` methods — so
this file is the ONLY place that needs to know which provider is behind it.
"""
import json
import os
import re

# Sonnet 4.6 is fast and plenty for claim decomposition (the verdict is the
# battery's job, not the LLM's). On-demand throughput isn't offered for the bare
# id, so we use the cross-region inference profile (`us.` prefix).
DEFAULT_BEDROCK_MODEL_ID = "us.anthropic.claude-sonnet-4-6"
ANTHROPIC_VERSION = "bedrock-2023-05-31"

# Terra balances quality/cost for structured JSON tasks (decompose/design/reflect);
# override with OPENAI_MODEL (e.g. "gpt-5.6-sol" for the flagship, "gpt-5.6-luna" for
# the cheapest/fastest tier) if you want a different tier.
DEFAULT_OPENAI_MODEL = "gpt-5.6-terra"


def _provider():
    return (os.environ.get("LLM_PROVIDER") or "openai").strip().lower()


def _region():
    return os.environ.get("BEDROCK_REGION") or os.environ.get("AWS_REGION") or "us-west-2"


def _model_id():
    return os.environ.get("BEDROCK_MODEL_ID") or DEFAULT_BEDROCK_MODEL_ID


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
    """Thin wrapper over an LLM provider (OpenAI or Bedrock). Never raises on construct.

    Kept named `ClaudeBedrock` for backward compatibility with every call site
    (`claims.py`, `probe_design.py`, `trace.py`, `dashboard/app_server.py`) — the name
    is now a misnomer for the default (OpenAI) path, but changing it would mean
    touching 4+ files for no functional benefit.
    """

    def __init__(self, model_id=None, region=None):
        self.provider = _provider()
        self._client = None      # openai.OpenAI, when provider == "openai"
        self._rt = None          # boto3 bedrock-runtime, when provider == "bedrock"
        self._err = None
        if os.environ.get("BEDROCK_DISABLE") or os.environ.get("LLM_DISABLE"):
            self._err = "LLM disabled via env"
            return
        if self.provider == "openai":
            self._init_openai(model_id)
        else:
            self._init_bedrock(model_id, region)

    def _init_openai(self, model_id):
        self.model_id = model_id or os.environ.get("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            self._err = ("OPENAI_API_KEY not set — get one at "
                         "https://platform.openai.com/api-keys (a ChatGPT Plus/Pro "
                         "subscription does not include API access; billing for API "
                         "usage is separate, set up at "
                         "https://platform.openai.com/account/billing)")
            return
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=key)
        except Exception as e:  # openai package missing / bad key format
            self._err = repr(e)

    def _init_bedrock(self, model_id, region):
        self.model_id = model_id or _model_id()
        self.region = region or _region()
        try:
            import boto3
            self._rt = boto3.client("bedrock-runtime", region_name=self.region)
        except Exception as e:  # boto3 missing / no creds resolvable
            self._err = repr(e)

    def available(self):
        return self._client is not None or self._rt is not None

    def _invoke(self, system, user, max_tokens=2000):
        if self._client is not None:
            return self._invoke_openai(system, user, max_tokens)
        return self._invoke_bedrock(system, user, max_tokens)

    def _invoke_openai(self, system, user, max_tokens):
        r = self._client.chat.completions.create(
            model=self.model_id, max_completion_tokens=max_tokens,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}])
        return r.choices[0].message.content or ""

    def _invoke_bedrock(self, system, user, max_tokens):
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
            "\"present\" or \"absent\"}. One element per distinct assertion. "
            "IMPORTANT: the substrate is TILE-LEVEL with NO notion of spatial position or "
            "adjacency. A claim whose testable content is spatial/positional — e.g. "
            "'peritumoral' / 'intratumoral' location, immune-inflamed vs -excluded vs "
            "-desert, tertiary lymphoid structures, tumor-vs-immune spatial relationship, "
            "perineural or lymphovascular invasion — has NO tile-level concept: set its "
            "concept to null so it is honestly declined. Keep the underlying PRESENCE claim "
            "separate (e.g. 'lymphocytic infiltrate present' maps to the immune concept; the "
            "'peritumoral' spatial qualifier is its own null-concept claim).")
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

    def narrate(self, compact):
        """One batched call: deterministic traces -> short plain-English glosses.

        Returns {"per_claim": {concept: sentence}, "overall": str}. The numbers and
        verdicts come from the deterministic trace; the LLM only phrases them — it must
        not invent or change any verdict.
        """
        if not self.available():
            raise RuntimeError(f"bedrock unavailable: {self._err}")
        system = (
            "You explain a causal-certification card to a pharma-governance reviewer. You "
            "are given deterministic per-claim reasoning traces (numbers + verdicts already "
            "computed). Phrase each claim's verdict in ONE plain sentence grounded in its "
            "trace, and write ONE overall sentence. Do NOT invent or change any verdict, "
            "score, or number. Output ONLY JSON: "
            "{\"per_claim\": {\"<concept>\": \"<sentence>\"}, \"overall\": \"<sentence>\"}.")
        user = json.dumps(compact)
        text = self._invoke(system, user, max_tokens=1200)
        start, end = text.find("{"), text.rfind("}")
        return json.loads(text[start:end + 1]) if start != -1 else {"overall": text.strip()}

    def propose_hypothesis(self, summary, max_tokens=1200):
        """Reflect on a certify card (score + reasoning trace) -> the NEXT hypothesis.

        Given the universal confidence, per-pillar verdicts, the deterministic reasoning
        trace, and the substrate's class vocabulary, propose (a) a diagnosis of which
        pillar is weak/strong and why, (b) the next causal hypothesis worth testing, (c) a
        concrete follow-up probe over the available classes, and (d) a short message to
        feed downstream to K-Pro or another Claude so the loop continues. Never decides
        certifiability — the deterministic battery re-checks any probe proposed. Raises on
        any failure so the caller can fall back to the deterministic heuristic.
        """
        if not self.available():
            raise RuntimeError(f"bedrock unavailable: {self._err}")
        system = (
            "You are a computational-pathology interpretability researcher running an "
            "ITERATIVE certify loop on a frozen pathology foundation model. You are given a "
            "causal-certification card: a universal confidence score in [0,1], per-pillar "
            "verdicts (necessity / sufficiency / specificity), a deterministic reasoning "
            "trace (the numbers behind the score), and the substrate's tissue-class "
            "vocabulary. Read the SCORE and the TRACE, diagnose which pillar is weakest and "
            "why, then propose the NEXT hypothesis to test and one concrete follow-up probe "
            "(a pos/neg contrast + a specificity distractor over ONLY the provided class "
            "codes) that would sharpen or falsify it. Prefer a tighter, matched foil over an "
            "easy one (never BACK/empty). Also write a short message to feed downstream to "
            "K-Pro (the pathology model) or another Claude so the loop continues. You NEVER "
            "decide certifiability — the deterministic battery re-checks any probe you "
            "propose. Output ONLY a JSON object: {\"diagnosis\": <one sentence grounded in "
            "the trace numbers>, \"weakest_pillar\": <\"necessity\"|\"sufficiency\"|"
            "\"specificity\">, \"next_hypothesis\": <one sentence>, \"proposed_probe\": "
            "{\"concept\": <snake_case>, \"pos\": <code>, \"neg\": <code>, \"distractor\": "
            "[<code>, <code>], \"rationale\": <one clause>}, \"message_to_downstream\": <one "
            "or two sentences>, \"feed_to\": <\"kpro\"|\"claude\">}.")
        user = json.dumps(summary)
        text = self._invoke(system, user, max_tokens=max_tokens)
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("no JSON object in model response")
        return json.loads(text[start:end + 1])
