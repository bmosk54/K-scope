"""Answer -> atomic, certifiable concept-claims.

This is the step that makes probing DYNAMIC and answer-bound instead of
taxonomy-bound. A K-Pro answer is free-form ("tumor epithelium with a brisk
peritumoral lymphocytic infiltrate, high-grade"); we split it into atomic claims
and resolve each to a *labeled contrast on this substrate*. Claims that don't
resolve to a labeled contrast are returned NOT_CERTIFIABLE rather than force-fit —
that is the "choose the prompts that would [read from H0/Phikon]" gate: the
scaffold only certifies claims the substrate has supervision for.

Decomposition is an LLM step (Claude on Bedrock) with a keyword-heuristic fallback
so the whole pipeline runs today with no Bedrock access.
"""
import re
from dataclasses import dataclass, field

from . import bedrock as _bedrock

# ---------------------------------------------------------------------------
# Concept vocabulary: free-text concept -> (pos, neg, distractor) over the
# substrate's labeled classes. This is the ONLY place taxonomy leaks in, and it
# is explicit + auditable. Abstract clinical claims (grade, MSI, "biomarker for
# X") deliberately have NO entry -> they resolve to NOT_CERTIFIABLE (needs labels).
# pos/neg/distractor are class-name strings that must exist in the track's classes.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ConceptSpec:
    concept: str
    pos: str
    neg: str
    distractor: tuple      # (pos, neg) orthogonal control
    keywords: tuple
    description: str


VOCAB = (
    ConceptSpec("tumor_epithelium", "TUM", "NORM", ("STR", "MUS"),
                ("tumor", "tumour", "carcinoma", "malignan", "adenocarcinoma", "neoplas"),
                "malignant epithelium vs normal mucosa"),
    ConceptSpec("immune_infiltrate", "LYM", "TUM", ("STR", "MUS"),
                ("lympho", "immune", "infiltrat", "til", "inflammat"),
                "lymphocytic/immune infiltrate vs tumor epithelium"),
    ConceptSpec("stroma", "STR", "MUS", ("TUM", "LYM"),
                ("stroma", "desmoplas", "fibro"),
                "cancer-associated stroma vs muscle"),
    ConceptSpec("mucus", "MUC", "NORM", ("STR", "MUS"),
                ("mucin", "mucus", "mucous"),
                "mucus vs normal mucosa"),
    ConceptSpec("muscle", "MUS", "STR", ("TUM", "LYM"),
                ("muscle", "muscularis", "smooth muscle"),
                "smooth muscle vs stroma"),
    ConceptSpec("adipose", "ADI", "NORM", ("STR", "MUS"),
                ("adipos", "fat tissue", "fatty"),
                "adipose vs normal mucosa"),
    ConceptSpec("debris_necrosis", "DEB", "NORM", ("STR", "MUS"),
                ("necros", "debris", "necrotic"),
                "debris/necrosis vs normal mucosa"),
    ConceptSpec("normal_mucosa", "NORM", "TUM", ("STR", "MUS"),
                ("normal mucosa", "benign", "non-neoplastic", "healthy"),
                "normal mucosa vs tumor"),
)

_BY_CONCEPT = {c.concept: c for c in VOCAB}


@dataclass
class Claim:
    text: str                       # the atomic phrase from the answer
    polarity: str = "present"       # "present" | "absent"
    spec: ConceptSpec = None        # resolved concept, or None if not certifiable
    status: str = "certifiable"     # "certifiable" | "not_certifiable"
    reason: str = ""                # why not, if not_certifiable
    meta: dict = field(default_factory=dict)

    @property
    def concept(self):
        return self.spec.concept if self.spec else None


def _resolvable_for(track):
    """Concepts whose pos/neg/distractor classes all exist on this track."""
    classes = set(track.class_names)
    return [c for c in VOCAB
            if {c.pos, c.neg, c.distractor[0], c.distractor[1]} <= classes]


def _resolve(concept_name, track):
    spec = _BY_CONCEPT.get(concept_name)
    if spec is None:
        return None
    classes = set(track.class_names)
    if not {spec.pos, spec.neg, spec.distractor[0], spec.distractor[1]} <= classes:
        return None  # concept known but not labeled on this substrate
    return spec


# ---------------------------------------------------------------------------
# Heuristic decomposition (fallback / offline path)
# ---------------------------------------------------------------------------
_CLAUSE_SPLIT = re.compile(r"[,;.]| with | and | showing | featuring ")
_ABSENT = re.compile(r"\b(no|absent|without|lack of|negative for)\b", re.I)


def _heuristic_decompose(answer, track):
    resolvable = {c.concept for c in _resolvable_for(track)}
    claims = []
    for clause in _CLAUSE_SPLIT.split(answer):
        clause = clause.strip()
        if not clause:
            continue
        polarity = "absent" if _ABSENT.search(clause) else "present"
        low = clause.lower()
        # Pick the concept with the MOST keyword hits, not the first in the list —
        # otherwise "peritumoral" (substring "tumor") wrongly beats "lymphocytic
        # infiltrate". Ties break on the longest matched keyword (more specific).
        scored = [(sum(k in low for k in c.keywords),
                   max((len(k) for k in c.keywords if k in low), default=0), c)
                  for c in VOCAB]
        hits, _, hit = max(scored, key=lambda s: (s[0], s[1]))
        if hits == 0:
            hit = None
        if hit is None:
            claims.append(Claim(text=clause, status="not_certifiable",
                                reason="no substrate-labeled concept matched"))
        elif hit.concept not in resolvable:
            claims.append(Claim(text=clause, polarity=polarity, status="not_certifiable",
                                reason=f"concept {hit.concept!r} not labeled on this track"))
        else:
            claims.append(Claim(text=clause, polarity=polarity, spec=hit))
    return claims


def decompose(answer, track, use_bedrock=True):
    """Split a K-Pro answer into atomic Claims resolved against this track.

    Tries Claude-on-Bedrock first (if creds present); falls back to the keyword
    heuristic. Both paths route every claim through the same certifiability gate,
    so an LLM cannot smuggle in a concept the substrate can't probe.
    """
    resolvable = _resolvable_for(track)
    if use_bedrock:
        client = _bedrock.ClaudeBedrock()
        if client.available():
            try:
                raw = client.decompose(answer, [c.concept for c in resolvable])
                out = []
                for r in raw:
                    spec = _resolve(r.get("concept"), track)
                    if spec is None:
                        out.append(Claim(text=r["text"], polarity=r.get("polarity", "present"),
                                         status="not_certifiable",
                                         reason="no substrate-labeled concept (LLM)"))
                    else:
                        out.append(Claim(text=r["text"], polarity=r.get("polarity", "present"),
                                         spec=spec, meta={"source": "bedrock"}))
                return out
            except Exception as e:
                # fall through to heuristic; record why on the first claim's meta
                claims = _heuristic_decompose(answer, track)
                if claims:
                    claims[0].meta["bedrock_error"] = repr(e)
                return claims
    return _heuristic_decompose(answer, track)
