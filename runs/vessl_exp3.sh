echo 0
python dbn.py \
    --config config_dsb/c10_frnrelu_AtoshABCnew.yaml \
    --feature_name feature.layer2stride1 \
    --tag DistoABC \
    --fat 1 \
    --optim_bs 128 \
    --joint 2 \
    --T 5 \
    --joint_depth 6 \
    --version v1.1.4 \
    --forget 0 \
    --beta1 0.0001 \
    --beta2 0.0001 \
    --linear_noise \
    --mixup_alpha 0.4 \
    --start_temp 2 \
    --input_scaling 2 \
    --ensemble_prediction 4 \
    --ensemble_exclude_a \
    --dsc \
    --frn_swish \
    --medium
echo 1
