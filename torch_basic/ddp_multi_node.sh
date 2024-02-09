#!/bin/bash

#SBATCH --job-name=ddp_multi_node
#SBATCH --time=00:30:00
#SBATCH --nodelist=hpe159,hpe160
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=16
#SBATCH --gpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --cpus-per-task=4
##SBATCH --account=ai1

##### Number of total processes 
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Nodelist:= " $SLURM_JOB_NODELIST
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "Ntasks per node:= "  $SLURM_NTASKS_PER_NODE
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

export LOGLEVEL=INFO
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_BLOCKING_WAIT=1
export NCCL_IB_DISABLE=1
##export NCCL_IB_HCA=mlx5
##export NCCL_SHM_DISABLE=1

echo "Run started at:- "
date

srun singularity exec --nv --nvccli -B /purestorage/:/purestorage /purestorage/slurm_test/pytorch_22.07-py3.sif \
torchrun \
--nnodes 2 \
--nproc_per_node 8 \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint 172.100.100.10:29500 \
/purestorage/slurm_test/7_multi-node-multi-gpu/train_ddp.py 5 exp1.yml


# /purestorage/slurm_test/7_multi-node-multi-gpu/multinode_torchrun.py 500 10
# python -m torch.distributed.launch --nnodes 2 --nproc_per_node 8 --master_addr 211.168.94.161 --master_port 29500 /purestorage/slurm_test/7_multi-node-multi-gpu/train_ddp.py 5 exp1.yml

# bash -c : 뒤의 문장을 command로 인식하라는 뜻. 리눅스 명령어를 동시에 쓰려고 (cd와 torchrun 동시에, && 앞에 실행이 되어야 뒤가 실행됨)
srun --container-image /purestorage/project/hkl/hkl_slurm/image_build/sqsh/ngctorch-2309-hkl-fr.sqsh \
--container-mounts /purestorage:/purestorage \
--no-container-mount-home \
--container-writable \
--container-workdir /purestorage/project/tyk \
bash -c "bash scripts/install.sh && \ 
torchrun \
--nnodes $nnode \
--nproc_per_node $SLURM_GPUS_PER_TASK \
--rdzv_id 2523525 \
--rdzv_backend static \
--master_addr $master_node \
--master_port 8882 \
--node_rank $RANK \
main.py --config config/hskang/ref_custom_convmixer.yaml"




srun --pty --nodelist=nv178 --gres=gpu:1 --container-image=docker://nvcr.io#nvidia/pytorch:23.04-py3 --container-name=pyxis_ngc2304 --job-name "diffusion" --container-mounts /purestorage:/purestorage  --container-writable /bin/bash




echo ""
echo "################################################################"
echo "@@@@@@@@@@@@@@@@@@ Run completed at:- @@@@@@@@@@@@@@@@@@@@@@@@@"
date
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo "################################################################"