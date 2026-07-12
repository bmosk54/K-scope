#!/usr/bin/env bash
# Grant the SageMaker execution role read access to the WSI slides.
#
# MUST be run by someone with IAM admin rights on account 509011742392 — the
# execution role itself cannot do this (a role can't grant itself privileges).
# After it runs, `python -m biolayer.data.wsi_thumbnail` works from the notebook.
#
#   ./deploy/grant_wsi_access.sh AmazonSageMaker-ExecutionRole-20260711T200940
#
set -euo pipefail

ROLE="${1:-AmazonSageMaker-ExecutionRole-20260711T200940}"
POLICY_FILE="$(dirname "$0")/wsi_read_policy.json"

echo "Attaching inline policy BiolayerWSIRead to role: $ROLE"
aws iam put-role-policy \
  --role-name "$ROLE" \
  --policy-name BiolayerWSIRead \
  --policy-document "file://${POLICY_FILE}"

echo "Done. Verify with:"
echo "  aws iam get-role-policy --role-name $ROLE --policy-name BiolayerWSIRead"
echo
echo "NOTE: if s3://bucketbiolayer is SSE-KMS encrypted, also grant kms:Decrypt"
echo "on the bucket's CMK to this role, or GetObject will still 403."
