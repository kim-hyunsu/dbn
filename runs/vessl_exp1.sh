echo 0
python dsb_renewal_fat.py \
    --config config_dsb/c10_frnrelu_AtoshABCnew.yaml \
    --feature_name feature.layer2stride1 \
    --tag AtoABC \
    --fat 3 \
    --optim_bs 128 \
    --joint 2 \
    --T 5 \
    --joint_depth 6 \
    --version v1.1.4 \
    --forget 1 \
    --beta1 0.0001 \
    --beta2 0.0001 \
    --linear_noise \
    --perturb 0.0001 \
    --perturb_mode 3
echo 1
python dsb_renewal_fat.py \
    --config config_dsb/c10_frnrelu_AtoshABCnew.yaml \
    --feature_name feature.layer2stride1 \
    --tag AtoABC \
    --fat 3 \
    --optim_bs 128 \
    --joint 2 \
    --T 5 \
    --joint_depth 6 \
    --version v1.1.4 \
    --forget 1 \
    --beta1 0.0001 \
    --beta2 0.0001 \
    --linear_noise \
    --perturb 0.00001 \
    --perturb_mode 3
echo 2