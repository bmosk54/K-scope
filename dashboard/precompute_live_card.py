"""Precompute the LIVE certify card once (GPU) and cache it for the dashboard.

Running the live source-intervention per request is too heavy for the UI, so we build
it once here and write the adapted window.CARD to public/live_card.json. build_all()
prefers that cached card when present, so the dashboard serves genuine live-necessity
verdicts with no per-request GPU cost.

    python dashboard/precompute_live_card.py [--per-class 10] [--n-null 12]
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import bridge
from live_slide import build_live_ctx

OUT = os.path.join(HERE, "public", "live_card.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=10)
    ap.add_argument("--n-null", type=int, default=12)
    ap.add_argument("--bedrock", action="store_true")
    a = ap.parse_args()

    print(f"[live] building slide (per_class={a.per_class}) …", flush=True)
    ctx = build_live_ctx(per_class=a.per_class, n_null=a.n_null)
    print(f"[live] slide={ctx['n_tiles']} tiles, classes={ctx['classes_present']}", flush=True)

    print("[live] running certify_answer with live_ctx (GPU) …", flush=True)
    card = bridge.build_card(use_bedrock=a.bedrock, fast=False, live_ctx=ctx)

    print(f"[live] intervened_on_input={card.get('_live')}", flush=True)
    for c in card["claims"]:
        if c.get("scores"):
            ln = "live" if c.get("live_necessity") else "cached"
            print(f"   {c['concept']:18s} V={c['verdict']:8s} nec={c['scores'].get('necessity')} "
                  f"({ln}) nec_cap={c.get('necessity_capped')}", flush=True)

    with open(OUT, "w") as f:
        json.dump(card, f, default=str)
    print(f"[live] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
