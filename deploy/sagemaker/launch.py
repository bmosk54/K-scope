"""CLI launcher — submit entry.py as a SageMaker Training Job on ml.g5.2xlarge.

No console/UI. Run from a terminal:

    export SAGEMAKER_ROLE_ARN=arn:aws:iam::735570134926:role/<execution-role>
    export HF_TOKEN=hf_xxx                       # for the gated H-optimus-0 download
    python deploy/sagemaker/launch.py            # add --pretrained 0 for a no-HF smoke test

Needs `pip install sagemaker`. The SDK packages this dir, uploads to the default
SageMaker bucket, resolves the PyTorch GPU DLC image, and starts the job — all via API.

Raw-CLI equivalent (if you'd rather not use the SDK): `aws sagemaker create-training-job`
with the same RoleArn, a PyTorch DLC image URI, ResourceConfig InstanceType=ml.g5.2xlarge,
and your source packaged to S3. The SDK just removes that boilerplate.
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", default=os.environ.get("SAGEMAKER_ROLE_ARN"))
    ap.add_argument("--instance-type", default="ml.g5.2xlarge")
    ap.add_argument("--pretrained", type=int, default=1)
    ap.add_argument("--framework-version", default="2.3")
    ap.add_argument("--py-version", default="py311")
    ap.add_argument("--region", default=os.environ.get("AWS_DEFAULT_REGION", "us-west-2"))
    ap.add_argument("--wait", action="store_true", help="stream logs until the job ends")
    args = ap.parse_args()

    if not args.role:
        sys.exit("ERROR: no execution role. Set SAGEMAKER_ROLE_ARN or pass --role. "
                 "The account has no passable SageMaker execution role yet — ask the "
                 "organizers for one (or create it, see deploy/sagemaker/README.md).")

    from sagemaker.pytorch import PyTorch

    env = {}
    if os.environ.get("HF_TOKEN"):
        env["HF_TOKEN"] = os.environ["HF_TOKEN"]

    est = PyTorch(
        entry_point="entry.py",
        source_dir=os.path.dirname(os.path.abspath(__file__)),
        role=args.role,
        instance_type=args.instance_type,
        instance_count=1,
        framework_version=args.framework_version,
        py_version=args.py_version,
        environment=env,
        hyperparameters={"pretrained": args.pretrained},
        base_job_name="hoptimus-edit",
        # requirements.txt in source_dir is auto-installed in the container.
    )
    est.fit(wait=args.wait)
    print("Submitted. Track: aws sagemaker describe-training-job --training-job-name",
          est.latest_training_job.name if est.latest_training_job else "<name>")


if __name__ == "__main__":
    main()
