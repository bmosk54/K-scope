"""Answer -> atomic, resolved concept-claims.

The step that makes probing DYNAMIC and answer-bound instead of taxonomy-bound. A
K-Pro answer is free-form ("tumor epithelium with a brisk peritumoral lymphocytic
infiltrate, high-grade"); we split it into atomic claims and resolve each against
the concept registry (concepts.py) — tissue labels AND HistoPLUS cell types. Each
claim carries the substrate + dataset it resolved to; claims with no substrate
label, or whose label source has no embeddings yet, are NOT_CERTIFIABLE (never
force-fit).

Decomposition is an LLM step (Claude on Bedrock) with a keyword-heuristic fallback,
so the pipeline runs with or without Bedrock. Both paths route every claim through
the same registry resolution.
"""
import re
from dataclasses import dataclass, field

from . import bedrock as _bedrock
from . import concepts as _concepts

_CLAUSE_SPLIT = re.compile(r"[,;.]| with | and | showing | featuring ")
_ABSENT = re.compile(r"\b(no|absent|without|lack of|negative for)\b", re.I)


@dataclass
class Claim:
    text: str
    polarity: str = "present"
    spec: object = None                 # concepts.ConceptSpec or None
    model_key: str = None               # substrate it resolved to
    dataset_slug: str = None            # label source it resolved to
    status: str = "certifiable"         # "certifiable" | "not_certifiable"
    reason: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def concept(self):
        return self.spec.concept if self.spec else None


def _to_claim(text, polarity, concept_name, preferred_model_key, split, source_tag):
    r = _concepts.resolve(concept_name, preferred_model_key, split)
    c = Claim(text=text, polarity=polarity, spec=r.spec, model_key=r.model_key,
              dataset_slug=r.dataset_slug, status=r.status, reason=r.reason)
    c.meta["decomposed_by"] = source_tag
    return c


def _heuristic_decompose(answer, preferred_model_key, split):
    claims = []
    for clause in _CLAUSE_SPLIT.split(answer):
        clause = clause.strip()
        if not clause:
            continue
        polarity = "absent" if _ABSENT.search(clause) else "present"
        matches = _concepts.match(clause)
        # Prefer a match that certifies on the preferred substrate; else best match.
        chosen = None
        for m in matches:
            if _concepts.resolve(m.concept, preferred_model_key, split).status == "certifiable":
                chosen = m.concept
                break
        if chosen is None:
            chosen = matches[0].concept if matches else None
        claims.append(_to_claim(clause, polarity, chosen, preferred_model_key,
                                split, "heuristic"))
    return claims


def decompose(answer, preferred_model_key="phikon_v2", split="train", use_bedrock=True):
    """Split a K-Pro answer into atomic Claims resolved against the registry.

    `preferred_model_key` is the substrate the caller would like to certify on
    (tissue concepts honor it; cell-type concepts always resolve to H0-mini). The
    LLM vocabulary is the FULL concept set — resolution, not the LLM, decides
    certifiability, so a model can't smuggle in an unprobeable concept.
    """
    vocab = [c.concept for c in _concepts.CONCEPTS]
    if use_bedrock:
        client = _bedrock.ClaudeBedrock()
        if client.available():
            try:
                raw = client.decompose(answer, vocab)
                return [_to_claim(r["text"], r.get("polarity", "present"),
                                  r.get("concept"), preferred_model_key, split, "bedrock")
                        for r in raw]
            except Exception as e:
                claims = _heuristic_decompose(answer, preferred_model_key, split)
                if claims:
                    claims[0].meta["bedrock_error"] = repr(e)
                return claims
    return _heuristic_decompose(answer, preferred_model_key, split)
