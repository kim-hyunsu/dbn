echo 0
python dbn.py \
    --config config_dsb/c10_frnrelu_AtoshABCnew.yaml \
    --feature_name feature.layer2stride1 \
    --tag AtoABC \
    --fat 1 \
    --optim_bs 128 \
    --T 5 \
    --joint_depth 6 \
    --version v1.1.4 \
    --forget 0 \
    --beta1 0.0001 \
    --beta2 0.0001 \
    --linear_noise \
    --mixup_alpha 0.4 \
    --start_temp 4 \
    --ensemble_prediction 3 \
    --dsc
echo 1
python naive_ed.py \
    --config config_naive_ed/c100_frnrelu_shX_t3.yaml \
    --seed 826791842 \
    --save ./checkpoints/naive_ed/c100/t235/230919/0
echo 2
python naive_ed.py \
    --config config_naive_ed/c100_frnrelu_shX_t3.yaml \
    --seed 18980201553 \
    --save ./checkpoints/naive_ed/c100/t235/230919/1
echo 3
python naive_ed.py \
    --config config_naive_ed/c100_frnrelu_shX_t3.yaml \
    --seed 27502822875 \
    --save ./checkpoints/naive_ed/c100/t235/230919/2
echo 4
python naive_ed.py \
    --config config_naive_ed/c100_frnrelu_shX_t3.yaml \
    --seed 4014015115 \
    --save ./checkpoints/naive_ed/c100/t235/230919/3
echo 5
python naive_ed.py \
    --config config_naive_ed/c100_frnrelu_shX_t3.yaml \
    --seed 3863321182 \
    --save ./checkpoints/naive_ed/c100/t235/230919/4
echo 6