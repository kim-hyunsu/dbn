# python dsb.py --logdir="eval/feature_last/example1" --lr_schedule="constant" --optim_ne=1000 --dataset="cifar10_feature" --beta1=1.5e-4 --beta2=1.5e-3 1>"dump/stdout1" 2>"dump/tfout1"
# python dsb.py --logdir="eval/feature_last/example2" --lr_schedule="constant" --optim_ne=1000 --dataset="cifar10_feature" --beta1=5e-4 --beta2=5e-3     1>"dump/stdout2" 2>"dump/tfout2"
# python dsb.py --logdir="eval/feature_last/example3" --lr_schedule="constant" --optim_ne=1000 --dataset="cifar10_feature" --beta1=2e-3 --beta2=2e-2     1>"dump/stdout3" 2>"dump/tfout3"
# python dsb.py --logdir="eval/feature_logit/example31" --lr_schedule="constant" --optim_ne=1000 --dataset="cifar10_logit" --features_dir='features' --beta1=1.5e-4 --beta2=1.5e-3 1>"dump/stdout31" 2>"dump/tfout31"
# mv dump/stdout31 eval/feature_logit/example31/stdout31
# mv dump/tfout31 eval/feature_logit/example31/tfout31
# python dsb.py --logdir="eval/feature_logit/example32" --lr_schedule="constant" --optim_ne=1000 --dataset="cifar10_logit" --features_dir='features' --beta1=5e-4 --beta2=5e-3     1>"dump/stdout32" 2>"dump/tfout32"
# mv dump/stdout32 eval/feature_logit/example32/stdout32
# mv dump/tfout32 eval/feature_logit/example32/tfout32

SUFFIX="resfeature_2e-3_2e-2_"
python dsb.py --logdir="eval/feature_logit/example"$SUFFIX --lr_schedule="constant" --optim_ne=1000 --dataset="cifar10_logit" --features_dir='features' --beta1=2e-3 --beta2=2e-2 --network='resnet' --version='v1.0'    1>dump/stdout$SUFFIX 2>dump/tfout$SUFFIX
mv dump/stdout$SUFFIX eval/feature_logit/example$SUFFIX/stdout$SUFFIX
mv dump/tfout$SUFFIX eval/feature_logit/example$SUFFIX/tfout$SUFFIX


# SUFFIX="resfeature_2e-2_2e-1_debug2"
# python dsb.py --logdir="eval/feature_logit/example"$SUFFIX --lr_schedule="constant" --optim_ne=1000 --dataset="cifar10_logit" --features_dir='features' --beta1=2e-2 --beta2=2e-1 --network='resnet' --version='v1.0'    1>dump/stdout$SUFFIX 2>dump/tfout$SUFFIX
# mv dump/stdout$SUFFIX eval/feature_logit/example$SUFFIX/stdout$SUFFIX
# mv dump/tfout$SUFFIX eval/feature_logit/example$SUFFIX/tfout$SUFFIX
