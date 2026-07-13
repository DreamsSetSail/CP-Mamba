gpu=2
port=$1  # 1234
dataset=pascal
exp_name=split$2  # 0/1/2/3
arch=HMNetAMP_Reg_MSFM_GatedRecap_Proto
net=$3  # vgg/renet50
postfix=$4  # manet/manet_5s

exp_dir=exp/${dataset}/${arch}/${exp_name}/${net}
snapshot_dir=${exp_dir}/snapshot
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}_${net}_${postfix}.yaml
mkdir -p ${snapshot_dir} ${result_dir}
now=$(date +"%Y%m%d_%H%M%S")
cp train_pascal_proto.sh train_pascal_proto.py ${config} ${exp_dir}

echo ${arch}
echo ${config}
export NCCL_P2P_DISABLE=1  # 某些主板/驱动兼容性问题
export NCCL_IB_DISABLE=1   # 如果不涉及跨机器集群，可以关掉 IB
export CUDA_VISIBLE_DEVICES=0,1
python3 -m torch.distributed.launch --nproc_per_node=${gpu} --master_port=${port} train_pascal_proto.py \
        --config=${config} \
        --arch=${arch} \
        --opts dynamic_proto_mode C  \
        2>&1 | tee ${result_dir}/train-$now.log
