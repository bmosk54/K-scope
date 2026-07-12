"""Agent probe design — let the LLM WRITE the causal probes, not just route to them.

The static path (concepts.py) maps an answer onto hand-authored ConceptSpecs whose
(pos, neg, distractor) contrasts a human picked. This module tests the next capability:
given a QUESTION and the substrate's raw class vocabulary, have Claude *design* the
contrasts itself — choose which classes form each concept's positive pool, negative
foil, and specificity distractor — and then let the SAME validation gate + causal
battery decide whether each proposed probe is a real causal axis.

The LLM never sees the certificate math and never decides certifiability. It only
proposes contrasts over the available labels; contrast validation (held-out AUROC,
intensity collinearity) and the battery (necessity / sufficiency / specificity vs a
matched-random null) keep an ill-posed contrast from being certified. That gate is
exactly what makes it safe to let an agent design probes.
"""
import json

from . import bedrock as _bedrock

# Standard NCT-CRC-HE class glosses. We give the model the biology of each code but
# NOT which contrasts to form — the pos/neg/distractor choice is the thing under test.
CLASS_GLOSS = {
    "ADI": "adipose (fat) tissue",
    "BACK": "background / empty slide (no tissue)",
    "DEB": "debris and necrosis",
    "LYM": "lymphocytes / immune infiltrate",
    "MUC": "mucus / mucin pools",
    "MUS": "smooth muscle (muscularis)",
    "NORM": "normal colonic mucosa (benign epithelium)",
    "STR": "cancer-associated stroma",
    "TUM": "colorectal adenocarcinoma epithelium (tumor)",
}

_SYSTEM = (
    "You are a computational pathologist designing LINEAR CONCEPT PROBES on a frozen "
    "pathology foundation-model's tile embeddings. A probe is a binary contrast over "
    "tissue classes that isolates ONE biological concept as a causal axis. For a good "
    "causal probe: (1) pos and neg must differ in the target concept but be as matched "
    "as possible on everything else (a tight foil, not an easy one like background); "
    "(2) the distractor is a DIFFERENT concept pair used to test specificity — pick two "
    "classes whose axis should be orthogonal to the concept; (3) avoid degenerate "
    "contrasts (BACK/empty as a foil separates 'tissue vs no-tissue', not biology). "
    "Output ONLY a JSON array. Each element: {\"concept\": <snake_case name>, "
    "\"pos\": <class code>, \"neg\": <class code>, \"distractor\": [<code>, <code>], "
    "\"rationale\": <one clause>}. Use ONLY the provided class codes.")


def design_probes(question, class_names, max_probes=8):
    """Ask Claude to design a probe battery answering `question` over `class_names`.

    Returns a list of {concept, pos, neg, distractor, rationale} dicts, filtered to
    contrasts whose classes all exist and where pos != neg. Raises if Bedrock is
    unavailable (caller decides whether to fall back).
    """
    client = _bedrock.ClaudeBedrock()
    if not client.available():
        raise RuntimeError(f"bedrock unavailable: {client._err}")
    classes = list(class_names)
    gloss = "\n".join(f"  {c}: {CLASS_GLOSS.get(c, '?')}" for c in classes)
    user = (f"Question to answer: {question!r}\n\n"
            f"Available tissue classes (use these codes ONLY):\n{gloss}\n\n"
            f"Design up to {max_probes} probes whose combined verdicts answer the "
            f"question. JSON array:")
    text = client._invoke(_SYSTEM, user, max_tokens=1500)
    raw = _extract_json_array(text)
    out = []
    cset = set(classes)
    for r in raw:
        pos, neg = r.get("pos"), r.get("neg")
        distr = tuple(r.get("distractor", []) or [])
        if pos not in cset or neg not in cset or pos == neg:
            continue
        distr = tuple(d for d in distr if d in cset)[:2]
        out.append({"concept": r.get("concept", f"{pos}_vs_{neg}"),
                    "pos": pos, "neg": neg,
                    "distractor": distr if len(distr) == 2 else ("STR", "MUS"),
                    "rationale": r.get("rationale", "")})
    return out


def _extract_json_array(text):
    import re
    fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.S)
    if fenced:
        return json.loads(fenced.group(1))
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end + 1])
    raise ValueError("no JSON array in model response")
