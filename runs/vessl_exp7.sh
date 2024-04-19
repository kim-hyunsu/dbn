echo 0
python dbn_tidy.py \
    --config config_dsb/c100_frnrelu_AtoABC_ensemble.yaml \
    --save ./checkpoints/dbn/c100/t111315/230925/v118/1 \
    --seed 682651
echo 1
python dbn_tidy.py \
    --config config_dsb/c100_frnrelu_AtoABC_ensemble.yaml \
    --save ./checkpoints/dbn/c100/t1123/230927/v118/1 \
    --tag AtoABC2 \
    --seed 480192
echo 2
python dbn_tidy.py \
    --config config_dsb/c100_frnrelu_AtoABC_distill.yaml \
    --distill ./checkpoints/dbn/c100/t111315/230925/v118/1 \
    --save ./checkpoints/dbn/c100/t111315dist/230925/v118/1 \
    --seed 175942
echo 3
python dbn_tidy.py \
    --config config_dsb/c100_frnrelu_AtoABC_distill.yaml \
    --distill ./checkpoints/dbn/c100/t1123/230927/v118/1 \
    --save ./checkpoints/dbn/c100/t1123dist/230927/v118/1 \
    --tag AtoABC2 \
    --seed 279856
echo 4
python dbn_tidy.py \
    --config config_dsb/c100_frnrelu_AtoABC_ensemble.yaml \
    --save ./checkpoints/dbn/c100/t1123/230927/v118/2 \
    --tag AtoABC2 \
    --seed 592361
echo 5
python dbn_tidy.py \
    --config config_dsb/c100_frnrelu_AtoABC_distill.yaml \
    --distill ./checkpoints/dbn/c100/t1123/230927/v118/2 \
    --save ./checkpoints/dbn/c100/t1123dist/230927/v118/2 \
    --tag AtoABC2 \
    --seed 530335
echo 6
python dbn_tidy.py \
    --config config_dsb/c100_frnrelu_AtoABC_ensemble.yaml \
    --save ./checkpoints/dbn/c100/t1123/230927/v118/3 \
    --tag AtoABC2 \
    --seed 127790
echo 7
python dbn_tidy.py \
    --config config_dsb/c100_frnrelu_AtoABC_distill.yaml \
    --distill ./checkpoints/dbn/c100/t1123/230927/v118/3 \
    --save ./checkpoints/dbn/c100/t1123dist/230927/v118/3 \
    --tag AtoABC2 \
    --seed 13021
echo 8
python dbn_tidy.py \
    --config config_dsb/c100_frnrelu_AtoABC_ensemble.yaml \
    --save ./checkpoints/dbn/c100/t1123/230927/v118/4 \
    --tag AtoABC2 \
    --seed 142564
echo 9
python dbn_tidy.py \
    --config config_dsb/c100_frnrelu_AtoABC_distill.yaml \
    --distill ./checkpoints/dbn/c100/t1123/230927/v118/4 \
    --save ./checkpoints/dbn/c100/t1123dist/230927/v118/4 \
    --tag AtoABC2 \
    --seed 174446
echo 10