echo 1
python proxy_end2.py \
    --config config_proxy_end2/c10_frnrelu_shX_t3.yaml \
    --seed 4186388129
echo 2
python proxy_end2.py \
    --config config_proxy_end2/c10_frnrelu_shX_t3.yaml \
    --seed 1299643586
echo 3
python proxy_end2.py \
    --config config_proxy_end2/c10_frnrelu_shX_t3.yaml \
    --seed 2771126769
echo 4
python proxy_end2.py \
    --config config_proxy_end2/c10_frnrelu_shX_t3.yaml \
    --seed 3153749052
echo 5
python proxy_end2.py \
    --config config_proxy_end2/c10_frnrelu_shX_t3.yaml
echo 6

