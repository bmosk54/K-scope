# Running K-Scope permanently, from your own laptop — no AWS required

The hackathon demo ran on a **temporary SageMaker Studio box** (workshop-issued AWS
credentials that expire in ~1h) and exposed the dashboard through a **Cloudflare quick
tunnel** (`dashboard/tunnel.sh`) — a random `https://<random>.trycloudflare.com` URL
that dies the moment that process or that box goes away. That's why the link you had
stopped working.

This doc replaces both of those with things you own: a local copy of the backend that
runs on your laptop whenever you want it, and (optionally) a permanent URL that keeps
pointing at it.

## What was actually AWS-dependent, and what wasn't

Good news first: most of the "AWS SageMaker + S3" setup was never load-bearing for
*running* the demo — it was there so a **team** could share one GPU box and one bucket
of pre-computed embeddings during the hackathon. Concretely:

| Piece | Was AWS for | Now, locally |
|---|---|---|
| GPU compute (torch forward passes) | SageMaker `ml.g5.2xlarge` (shared, ephemeral) | Your laptop's CPU (or Apple Silicon MPS) — phikon-v2 is a ViT-L, small enough to run comfortably without a GPU. `biolayer/data/models.py` already auto-selects CUDA → MPS → CPU. |
| Embeddings cache (`embeddings/…npz`) | `s3://bucketbiolayer` | `artifacts/` on disk. **`biolayer/data/loader.py` already prefers the local `artifacts/` folder and only falls back to S3** — so once you extract once locally, S3 is never touched. |
| Frozen model weights (phikon-v2) | downloaded via HuggingFace, cached wherever the box's `~/.cache/huggingface` was | your laptop's `~/.cache/huggingface` — same mechanism, just persists locally now. phikon-v2 is **ungated**, so no HF login is even needed. |
| Public URL | Cloudflare **quick** tunnel (`dashboard/tunnel.sh`) — random subdomain, dies with the process | Cloudflare **named** tunnel (`setup_cloudflare_tunnel.sh`, this folder) — one fixed hostname you own, reusable forever. |
| K-Pro answer / prompt-optimizer LLM calls | AWS Bedrock (Claude), auth via the SageMaker execution role | your own OpenAI API key (see below) — swapped in `biolayer/dynamic/bedrock.py`. |
| Slide Gallery tab images | `s3://bucketbiolayer/galleries/…` (`dashboard/fetch_galleries.py`) | still S3-only; this is the one tab that stays empty without AWS creds. Everything else (Case/Proof/Verdict/AutoResearch/Evidence) doesn't touch it. It already degrades gracefully (best-effort fetch, warns and continues). |

So: **certify / steer / ablate / probe / specificity / confound / layered / AutoResearch
loop all run on pure local torch + numpy, no AWS account, ever**, once you've done the
one-time local extraction below.

## One-time setup

```bash
# 1. everything: venv, deps, local phikon-v2 embeddings extracted to artifacts/
bash deploy/local/bootstrap_local.sh

# 2. your OpenAI key for the two LLM-backed buttons (optional — see note below)
cp deploy/local/env.example .env.local
$EDITOR .env.local        # paste OPENAI_API_KEY

# 3. optional: a permanent public URL instead of localhost-only
bash deploy/local/setup_cloudflare_tunnel.sh kscope.yourdomain.com
```

Step 1 downloads torch/transformers (~2-3 GB) and then streams ~2,700 H&E tiles from
the `1aurent/NCT-CRC-HE` HuggingFace dataset and embeds them with phikon-v2 — this is
the one-time replacement for pulling the pre-computed `.npz` from
`s3://bucketbiolayer`. It's CPU-only-friendly; expect low-single-digit minutes, not
hours. Re-run with `PER_CLASS=800 bash deploy/local/bootstrap_local.sh` later for a
tighter probe fit if you want (more tiles = better statistics, slower extraction).

Step 3 needs a **free Cloudflare account** and **a domain added to Cloudflare** (any
domain from any registrar — Cloudflare just needs to manage its DNS, for free). If you
don't have a domain, skip step 3 entirely: `bin/kscope start` still works, just
serving `http://localhost:4173` instead of a public URL.

### About the OpenAI key

The dashboard's **"K-Pro answer"** and **"optimize prompt"** buttons call an LLM to (a)
draft a plausible answer from the tile and (b) rewrite your question. That's it — the
actual certification (necessity/sufficiency/specificity vs matched-random null) is
deterministic torch/numpy and never calls an LLM.

You mentioned you have ChatGPT **Pro**. Important: **the ChatGPT Plus/Pro subscription
does not include API access.** They're two separate products with separate billing:

- ChatGPT Plus/Pro = the chat.openai.com web/app product you're paying for monthly.
- OpenAI **API** = a pay-per-token developer product, billed separately, with its own
  key from <https://platform.openai.com/api-keys> and its own billing page at
  <https://platform.openai.com/account/billing>.

So: create an API key there, add a few dollars of credit (this demo costs fractions of
a cent per call), and put it in `.env.local` as `OPENAI_API_KEY`. If you'd rather not
do that, leave it blank — everything except those two buttons still works
(`bin/kscope start` will just show "bedrock unavailable" / a 503 on those two routes;
the rest of the evidence card is unaffected).

## Day-to-day use

You said you don't want this running all the time (fair — torch + a warm Flask
process is not nothing). So nothing auto-starts on login; you start it when you want
it and stop it when you're done:

```bash
bin/kscope start      # backend (+ tunnel, if configured) come up; prints the URL
bin/kscope status     # is it running? what's the URL?
bin/kscope stop       # everything goes down, nothing left running
bin/kscope logs       # tail the backend + tunnel logs
```

First `start` after a reboot takes a few extra seconds (loading torch + the phikon-v2
weights into memory); subsequent requests are fast because `dashboard/app_server.py`
is a warm, resident process (see its module docstring) — that part of the design
didn't change.

If you ever DO want it always-on and auto-restarting (e.g. you decide to leave a home
server running this instead of your laptop), the natural next step is a macOS
`launchd` agent wrapping `bin/kscope start` with `KeepAlive`/`RunAtLoad` — ask for it
and it's a small addition, deliberately left out here since you said "not always".

## Why this doesn't need SageMaker or S3 at all going forward

`biolayer/config.py`, `biolayer/data/s3_utils.py` and `deploy/sagemaker/` are all still
there and still work if you (or teammates) want to go back to the shared-team AWS
setup — nothing was deleted. But nothing in the local path above imports or requires
them: `loader.py`'s local-artifacts-first behavior and `serving.py`'s local disk/RAM
caches were already written to degrade this way; this setup just exercises that path
exclusively instead of falling back to it.
