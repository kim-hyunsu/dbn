echo 0
python dsb_renewal_fat.py --config config_dsb/c10_frnrelu_AtoshABCnew.yaml \
    --feature_name feature.layer3stride2 \
    --tag AtoshABC \
    --fat 2 \
    --optim_bs 128 \
    --joint 2 \
    --forget \
    --T 2 \
    --kld_joint \
    --version v1.1
echo 1
