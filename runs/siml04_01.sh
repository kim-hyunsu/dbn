export CUDA_VISIBLE_DEVICES=0,1
echo 0
taskset -c 0-23 python dbn.py \
    --config config_dsb/c10_frnrelu_AtoABC_ensemble.yaml \
    --prob_loss
echo 1
taskset -c 0-23 python dbn.py \
    --config config_dsb/c10_frnrelu_AtoABC_ensemble.yaml \
    --kld_joint \
    --beta 1
echo 2
