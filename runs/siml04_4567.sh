echo 4
taskset -c 48-95 python scripts/train_bezier.py \
    -f configs/tin200/bezier.yaml \
    -ca outs-z/checkpoints/frn200_sd3_be/checkpoint_180/best_acc1.pt \
    -cb outs-z/checkpoints/frn200_sd5_be/checkpoint_175/best_acc1.pt \
    -o outs-z/checkpoints/t200/bezier/3-5/0
echo 5
taskset -c 48-95 python scripts/train_bezier.py \
    -f configs/tin200/bezier.yaml \
    -ca outs-z/checkpoints/frn200_sd3_be/checkpoint_180/best_acc1.pt \
    -cb outs-z/checkpoints/frn200_sd7_be/checkpoint_164/best_acc1.pt \
    -o outs-z/checkpoints/t200/bezier/3-7/0
echo 6