echo 0
python proxy_end2.py \
    --config config_proxy_end2/c10_frnrelu_t5.yaml \
    --seed 2249972020 \
    --save ./checkpoints/proxy_end2/c10/t23579/230923/0
echo 1
python proxy_end2.py \
    --config config_proxy_end2/c10_frnrelu_t5.yaml \
    --seed 497488857 \
    --save ./checkpoints/proxy_end2/c10/t23579/230923/1
echo 2
python proxy_end2.py \
    --config config_proxy_end2/c10_frnrelu_t5.yaml \
    --seed 2696417285 \
    --save ./checkpoints/proxy_end2/c10/t23579/230923/2
echo 3
python proxy_end2.py \
    --config config_proxy_end2/c10_frnrelu_t5.yaml \
    --seed 1349922039 \
    --save ./checkpoints/proxy_end2/c10/t23579/230923/3
echo 4
python proxy_end2.py \
    --config config_proxy_end2/c10_frnrelu_t5.yaml \
    --seed 2910230587 \
    --save ./checkpoints/proxy_end2/c10/t23579/230923/4
echo 5
python proxy_end2.py \
    --config config_proxy_end2/c10_frnrelu_t7.yaml \
    --seed 3914743490 \
    --save ./checkpoints/proxy_end2/c10/t235791113/230923/0
echo 6
python proxy_end2.py \
    --config config_proxy_end2/c10_frnrelu_t7.yaml \
    --seed 1587805547 \
    --save ./checkpoints/proxy_end2/c10/t235791113/230923/1
echo 7
python proxy_end2.py \
    --config config_proxy_end2/c10_frnrelu_t7.yaml \
    --seed 377502791 \
    --save ./checkpoints/proxy_end2/c10/t235791113/230923/2
echo 8
python proxy_end2.py \
    --config config_proxy_end2/c10_frnrelu_t7.yaml \
    --seed 2063208277 \
    --save ./checkpoints/proxy_end2/c10/t235791113/230923/3
echo 9
python proxy_end2.py \
    --config config_proxy_end2/c10_frnrelu_t7.yaml \
    --seed 761055307 \
    --save ./checkpoints/proxy_end2/c10/t235791113/230923/4
echo 10
python proxy_end2.py \
    --config config_proxy_end2/c10_frnrelu_t9.yaml \
    --seed 179086698 \
    --save ./checkpoints/proxy_end2/c10/t2357911131517/230923/0
echo 11
python proxy_end2.py \
    --config config_proxy_end2/c10_frnrelu_t9.yaml \
    --seed 1170717227 \
    --save ./checkpoints/proxy_end2/c10/t2357911131517/230923/1
echo 12
python proxy_end2.py \
    --config config_proxy_end2/c10_frnrelu_t9.yaml \
    --seed 2306825241 \
    --save ./checkpoints/proxy_end2/c10/t2357911131517/230923/2
echo 13
python proxy_end2.py \
    --config config_proxy_end2/c10_frnrelu_t9.yaml \
    --seed 757026008 \
    --save ./checkpoints/proxy_end2/c10/t2357911131517/230923/3
echo 14
python proxy_end2.py \
    --config config_proxy_end2/c10_frnrelu_t9.yaml \
    --seed 2178467949 \
    --save ./checkpoints/proxy_end2/c10/t2357911131517/230923/4
echo 15
python proxy_end2.py \
    --config config_proxy_end2/c100_frnrelu_t5.yaml \
    --seed 2249972020 \
    --save ./checkpoints/proxy_end2/c100/t11131523/230923/0
echo 16
python proxy_end2.py \
    --config config_proxy_end2/c100_frnrelu_t5.yaml \
    --seed 497488857 \
    --save ./checkpoints/proxy_end2/c100/t11131523/230923/1
echo 17
python proxy_end2.py \
    --config config_proxy_end2/c100_frnrelu_t5.yaml \
    --seed 2696417285 \
    --save ./checkpoints/proxy_end2/c100/t11131523/230923/2
echo 18
python proxy_end2.py \
    --config config_proxy_end2/c100_frnrelu_t5.yaml \
    --seed 1349922039 \
    --save ./checkpoints/proxy_end2/c100/t11131523/230923/3
echo 19
python proxy_end2.py \
    --config config_proxy_end2/c100_frnrelu_t5.yaml \
    --seed 2910230587 \
    --save ./checkpoints/proxy_end2/c100/t11131523/230923/4
echo 20
python proxy_end2.py \
    --config config_proxy_end2/c100_frnrelu_t7.yaml \
    --seed 3914743490 \
    --save ./checkpoints/proxy_end2/c100/t1113152357/230923/0
echo 21
python proxy_end2.py \
    --config config_proxy_end2/c100_frnrelu_t7.yaml \
    --seed 1587805547 \
    --save ./checkpoints/proxy_end2/c100/t1113152357/230923/1
echo 22
python proxy_end2.py \
    --config config_proxy_end2/c100_frnrelu_t7.yaml \
    --seed 377502791 \
    --save ./checkpoints/proxy_end2/c100/t1113152357/230923/2
echo 23
python proxy_end2.py \
    --config config_proxy_end2/c100_frnrelu_t7.yaml \
    --seed 2063208277 \
    --save ./checkpoints/proxy_end2/c100/t1113152357/230923/3
echo 24
python proxy_end2.py \
    --config config_proxy_end2/c100_frnrelu_t7.yaml \
    --seed 761055307 \
    --save ./checkpoints/proxy_end2/c100/t1113152357/230923/4
echo 25
/mnt/home/hyunsu/.conda/envs/dsb/bin/wandb agent kim-hyunsu/dbn/lcwzkndb
