# Infra — running biolayer on a GPU via EKS

Cluster **`fabulous-pop-sculpture`** exists in **us-west-2**. Recommended path: a
**managed GPU nodegroup** (not a hand-attached EC2, not hybrid). Files here:

- [`eksctl-gpu-nodegroup.yaml`](eksctl-gpu-nodegroup.yaml) — managed `g5.2xlarge` nodegroup
- [`k8s/gpu-job.yaml`](k8s/gpu-job.yaml) — biolayer extractor Job requesting `nvidia.com/gpu: 1`
- [`eksctl-hybrid-cluster.yaml`](eksctl-hybrid-cluster.yaml) + [`hybrid-node/`](hybrid-node) — the hybrid-node *alternative* (kept for reference)

## ⚠️ Blocker first: GPU quota is 0

`Running On-Demand G and VT instances` = **0** in us-west-2 (Spot G also 0; no capacity
reservations — probed 2026-07-11). A `g5.2xlarge` consumes **8 G-family vCPUs**, so the
nodegroup's ASG will **fail to launch** with `VcpuLimitExceeded` regardless of how it's
created. **An admin must raise that quota to ≥ 8** (Service Quotas → EC2 →
"Running On-Demand G and VT instances"). `WSParticipantRole` cannot request this itself.

Until then, the only provisioned GPU on this account is a **SageMaker `ml.g5.2xlarge`**
(quota = 1, same A10G) — see [../docs/SETUP.md](../docs/SETUP.md).

## Steps (once quota ≥ 8)

1. **Create the nodegroup** — eksctl auto-creates the node IAM role, picks the AL2023
   **NVIDIA** accelerated AMI for GPU instance types, and wires CNI perms:
   ```bash
   eksctl create nodegroup -f infra/eksctl-gpu-nodegroup.yaml
   ```
   (Or the AWS Console flow: EKS → cluster → Compute → Add node group, AMI type
   **Amazon Linux 2023 x86_64 NVIDIA**, `g5.2xlarge`, disk 100 GiB, min 0 / desired 1 /
   max 1. Node role must trust `ec2.amazonaws.com` and carry `AmazonEKSWorkerNodePolicy`
   + `AmazonEC2ContainerRegistryPullOnly`; give the VPC CNI its perms via IRSA/Pod Identity.)

2. **Point kubectl at the cluster + confirm the node joined:**
   ```bash
   aws eks update-kubeconfig --region us-west-2 --name fabulous-pop-sculpture
   kubectl get nodes -o wide            # the g5 node should show Ready
   ```

3. **Install the NVIDIA device plugin** (AMI has drivers, not the k8s plugin):
   ```bash
   helm repo add nvdp https://nvidia.github.io/k8s-device-plugin && helm repo update
   helm upgrade --install nvdp nvdp/nvidia-device-plugin \
     --namespace nvidia --create-namespace --set gfd.enabled=true
   kubectl get nodes -o=custom-columns='NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu'
   ```

4. **Run the workload:**
   ```bash
   kubectl create secret generic hf-token --from-literal=token=<hf_read_token>
   kubectl apply -f infra/k8s/gpu-job.yaml      # set image + track first
   kubectl logs -f job/biolayer-extract
   ```

## Still to wire

- **Container image.** Build `torch`+`timm`+`biolayer` into an image and push to ECR
  (the node's role has `ECRContainerRegistryPullOnly`); set it in `gpu-job.yaml`.
- **S3 for artifacts.** Prefer IRSA on the Job's service account for `bucketbiolayer`
  (SETUP.md notes the current role is ListBucket-only — may need a policy fix to upload).
