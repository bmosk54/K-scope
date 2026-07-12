"""biolayer.dynamic — answer-bound dynamic-probe certification.

Re-binds certification from a fixed tissue taxonomy to the concepts a specific
K-Pro answer actually asserts. The causal machinery (source-intervention battery +
matched-random null + confound gate) is reused unchanged from biolayer.causal; the
new part is decomposing an answer into atomic, substrate-labeled claims and
certifying each one, with the null/specificity/confound/held-out/multiple-
comparisons guards enforced in the tool contract.

Entry point:
    from biolayer.dynamic import certify_answer
    card = certify_answer(prompt, answer, track="phikon")
"""
from .certify_answer import certify_answer

__all__ = ["certify_answer"]
