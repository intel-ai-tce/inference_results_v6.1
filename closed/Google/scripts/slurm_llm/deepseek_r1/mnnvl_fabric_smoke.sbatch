#!/bin/bash
#SBATCH --account=coreai_mlperf_inference
#SBATCH --partition=a4x
#SBATCH --job-name=dp-r1-mnnvl-smoke
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:10:00
#SBATCH --output=build/dp_r1_mnnvl_smoke_%j.out
#SBATCH --error=build/dp_r1_mnnvl_smoke_%j.err

set -euo pipefail

IMAGE=${IMAGE:-/lustre/alisachen/containers/mlperf-inference_tensorrt_llm_release-feat-1.2-mlpinf-b5ddff4_mlperf-main-f538816_jan28_aarch64.sqsh}
REPO=${REPO:-${SLURM_SUBMIT_DIR:-/home/alisachen_google_com/nv-mlpinf-partner/closed/NVIDIA}}
MPI_MODE=${MPI_MODE:-pmi2}

echo "container_image=${IMAGE}"
echo "mpi_mode=${MPI_MODE}"

srun --mpi="${MPI_MODE}" \
  --nodes=2 \
  --ntasks=8 \
  --ntasks-per-node=4 \
  --gpus-per-task=1 \
  --label \
  --container-image="${IMAGE}" \
  --container-mount-home \
  --container-remap-root \
  --container-workdir=/work \
  --container-mounts="${REPO}:/work" \
  python3 /work/scripts/slurm_llm/deepseek_r1/check_mnnvl_fabric.py

echo "PASS: MNNVL topology smoke test completed"
