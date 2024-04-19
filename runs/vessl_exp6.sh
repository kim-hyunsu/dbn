echo 0
python dbn_tidy.py \
    --config config_dsb/c100_frnrelu_AtoABC_ensemble.yaml \
    --save ./checkpoints/dbn/c100/t111315/230925/v118/0
echo 1
python dbn_tidy.py \
    --config config_dsb/c100_frnrelu_AtoABC_ensemble.yaml \
    --save ./checkpoints/dbn/c100/t1123/230927/v118/0 \
    --tag AtoABC2 \
    --seed 307413
echo 2
python dbn_tidy.py \
    --config config_dsb/c100_frnrelu_AtoABC_distill.yaml \
    --distill ./checkpoints/dbn/c100/t111315/230925/v118/0 \
    --save ./checkpoints/dbn/c100/t111315dist/230925/v118/0 \
    --seed 873747
echo 3
python dbn_tidy.py \
    --config config_dsb/c100_frnrelu_AtoABC_distill.yaml \
    --distill ./checkpoints/dbn/c100/t1123/230927/v118/0 \
    --save ./checkpoints/dbn/c100/t1123dist/230927/v118/0 \
    --tag AtoABC2 \
    --seed 210492
echo 4
python dbn_tidy.py \
    --config config_dsb/c100_frnrelu_AtoABC_ensemble.yaml \
    --save ./checkpoints/dbn/c100/t111315/230925/v118/2 \
    --seed 595823
echo 5
python dbn_tidy.py \
    --config config_dsb/c100_frnrelu_AtoABC_distill.yaml \
    --distill ./checkpoints/dbn/c100/t111315/230925/v118/2 \
    --save ./checkpoints/dbn/c100/t111315dist/230925/v118/2 \
    --seed 85995
echo 6
python dbn_tidy.py \
    --config config_dsb/c100_frnrelu_AtoABC_ensemble.yaml \
    --save ./checkpoints/dbn/c100/t111315/230925/v118/3 \
    --seed 749
echo 7
python dbn_tidy.py \
    --config config_dsb/c100_frnrelu_AtoABC_distill.yaml \
    --distill ./checkpoints/dbn/c100/t111315/230925/v118/3 \
    --save ./checkpoints/dbn/c100/t111315dist/230925/v118/3 \
    --seed 207223
echo 8
python dbn_tidy.py \
    --config config_dsb/c100_frnrelu_AtoABC_ensemble.yaml \
    --save ./checkpoints/dbn/c100/t111315/230925/v118/4 \
    --seed 845076
echo 9
python dbn_tidy.py \
    --config config_dsb/c100_frnrelu_AtoABC_distill.yaml \
    --distill ./checkpoints/dbn/c100/t111315/230925/v118/4 \
    --save ./checkpoints/dbn/c100/t111315dist/230925/v118/4 \
    --seed 857696
echo 10

