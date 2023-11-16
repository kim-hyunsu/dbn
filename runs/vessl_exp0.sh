echo 0
python dbn_tidy.py \
    --config config_dsb/c10_frnrelu_Gaussianprior.yaml \
    --save ./checkpoints/dbn/c10/gaussian/0 \
    --seed 111111
echo 1
python dbn_tidy.py \
    --config config_dsb/c10_frnrelu_Gaussianprior.yaml \
    --save ./checkpoints/dbn/c10/gaussian/1 \
    --seed 2222
echo 2
python dbn_tidy.py \
    --config config_dsb/c10_frnrelu_Gaussianprior.yaml \
    --save ./checkpoints/dbn/c10/gaussian/2 \
    --seed 333333
echo 3