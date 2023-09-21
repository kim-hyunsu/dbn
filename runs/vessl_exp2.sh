echo 0
python dbn.py \
    --config config_dsb/c100_frnrelu_AtoABC_ensemble.yaml \
    --save ./checkpoints/dbn/c100/t111315 \
    --seed 1277792997
echo 1
python dbn.py \
    --config config_dsb/c100_frnrelu_AtoABC_ensemble.yaml \
    --save ./checkpoints/dbn/c100/t111315 \
    --seed 146108749
echo 2
python dbn.py \
    --config config_dsb/c100_frnrelu_AtoABC_ensemble.yaml \
    --save ./checkpoints/dbn/c100/t111315 \
    --seed 3594380594
echo 3
