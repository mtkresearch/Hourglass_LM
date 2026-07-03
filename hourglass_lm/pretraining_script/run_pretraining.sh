
config_pth=$1
olmo_eval_config_pth=$2

# Change to your own path
megatron_bridge_home=ABSOLUTE_PATH_TO_MEGATRON_BRIDGE
work_dir=ABSOLUTE_PATH_TO_NEMO_DIR


export PYTHONPATH=$megatron_bridge_home/src:$megatron_bridge_home/3rdparty/Megatron-LM:$olmo_home:$PYTHONPATH

cd $work_dir/pretraining_script

random_num=$(( RANDOM % 10001 + 30000 ))

torchrun --nproc_per_node=1 --master_port=$random_num custom_mlp_pretraining_example.py \
    --config-file $config_pth
