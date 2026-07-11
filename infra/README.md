# Infra — running biolayer on GPU via EKS **Hybrid Nodes**

## Why hybrid, not a normal nodegroup

This workshop account has **0 GPU quota** — On-Demand *and* Spot G/VT vCPU = 0 in
us-west-2 and us-east-1, no capacity reservations (probed 2026-07-11). A normal EKS
managed nodegroup launches EC2 G-instances, which draw on that 0 quota and **fail at
node-join**. `WSParticipantRole` cannot request a quota increase.

**Hybrid Nodes** sidestep this: you attach an **external** GPU machine (your own box /
on-prem / another cloud) to the EKS control plane with `nodeadm`. AWS never launches an
EC2 instance for it, so the EC2 quota is irrelevant. The control plane runs in AWS; the
GPU is your hardware. (Fargate — the other non-EC2 EKS compute — has **no GPU support**,
so it can't be used here.)

## What you must supply

- **An external Linux GPU machine** with a supported OS (AL2023 / Ubuntu 20.04+ / RHEL 8+),
  an NVIDIA driver, and containerd. This repo's WSL2 box has **no GPU** — it can't be the
  node. Source a GPU box (personal workstation, lab machine, RunPod/Lambda/GCP instance, …).

## Steps

1. **Create the cluster** (control plane only, no EC2 workers):
   ```bash
   eksctl create cluster -f infra/eksctl-hybrid-cluster.yaml
   ```
   Edit the `remoteNodeNetworks` / `remotePodNetworks` CIDRs first to match your GPU box.

2. **Create node credentials (admin step).** Hybrid nodes authenticate via an SSM hybrid
   activation *or* IAM Roles Anywhere:
   ```bash
   aws ssm create-activation --region us-west-2 \
     --default-instance-name owkin-hybrid-gpu \
     --iam-role <EKSHybridNodeRole> --registration-limit 1
   ```
   Put the returned `ActivationCode` / `ActivationId` into
   [hybrid-node/nodeadm-config.yaml](hybrid-node/nodeadm-config.yaml).

3. **Install a hybrid-compatible CNI** (VPC CNI is not supported on hybrid nodes):
   ```bash
   helm install cilium cilium/cilium -n kube-system \
     --set ipam.mode=cluster-pool --set ipam.operator.clusterPoolIPv4PodCIDRList='{10.86.0.0/16}'
   ```

4. **Join the GPU node** (run on the external machine) — see the header of
   [hybrid-node/nodeadm-config.yaml](hybrid-node/nodeadm-config.yaml) for the `nodeadm`
   install + `init` commands. Then install the NVIDIA device plugin so pods can request GPUs:
   ```bash
   kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.15.0/deployments/static/nvidia-device-plugin.yml
   kubectl get nodes -o wide     # your GPU box should show Ready
   ```

5. **Run the workload:**
   ```bash
   kubectl create secret generic hf-token --from-literal=token=<hf_read_token>
   kubectl apply -f infra/k8s/gpu-job.yaml     # set image + track first
   kubectl logs -f job/biolayer-extract
   ```

## Feasibility flags — verify before committing time

These are the parts most likely to block a **participant** role; confirm with organizers:

- **Networking is the hard prerequisite.** The control plane must route to your node's
  node/pod CIDRs and vice-versa — normally **site-to-site VPN or Direct Connect**. Over
  plain internet this needs deliberate setup; `kubectl logs/exec`, webhooks, and metrics
  all depend on control-plane→node reachability.
- **IAM.** `ssm:CreateActivation` + creating the `EKSHybridNodeRole` (or an IAM Roles
  Anywhere trust anchor/profile) likely exceeds `WSParticipantRole` permissions. May need
  an admin/organizer to run step 2.
- **Cluster create.** `eksctl create cluster` provisions a VPC + control plane (billable,
  slow ~15 min) — confirm the participant role may create EKS clusters.

If any of these are blocked, the fallback that *is* provisioned on this account is a
**SageMaker `ml.g5.2xlarge`** (quota = 1) — same A10G GPU, no cluster/networking, per
[../docs/SETUP.md](../docs/SETUP.md).
