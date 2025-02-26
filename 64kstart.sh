
export TZ=UTC-8
export HCCL_IF_BASE_PORT=30000
export HCCL_CONNECT_TIMEOUT=1200
export HCCL_INTRA_ROCE_ENABLE=1
export HCCL_INTRA_PCIE_ENABLE=0
export HCCL_RDMA_TC=104
export HCCL_BUFFSIZE=200 
export INTERNLM_ACCELERATOR="ditorch"
export DEEPLINK_EXT_PLATFORM_TYPE=torch_npu
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LAUNCH_BLOCKING=1
export PYTHONPATH=/deeplink_afs/zhumingzhu/code/envs/rotary_emb_no_interleave/DeepLinkExt:/deeplink_afs/zhumingzhu/code/envs:$PYTHONPATH

source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /opt/InternEvo

log_file="llama2_7B_internevo_$(date +%Y%m%d_%H%M%S)_${RANK}"

torchrun --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --nproc_per_node=8 --nnodes=16 --node_rank=$RANK train.py --config configs/7B_llama2_longseq64k.py --launcher torch --seed 42  2>&1 | tee /deeplink_afs/wangqing/InternEvo/logs/${log_file}.log

