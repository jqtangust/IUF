# ./schedctl create --name our14 --image "harbor.smoa.cc/public/jiaqitang:HDR-v3" --gpu 6 --cmd "/dataset/ruizhengwu/SmallDefect/Final_UniAD_14_classpara_Vit_hyperin_s_emaold_u1000_batchmore_zv5_2_OursSGD/run.sh" --arch ampere

# export PYTHONPATH=../../:$PYTHONPATH
# pip install scikit-learn
# pip install tensorboardX
# pip install tabulate
# pip install /dataset/ruizhengwu/SmallDefect/torch-1.11.0+cu113-cp38-cp38-linux_x86_64.whl
cd /dataset/ruizhengwu/SmallDefect/Final_UniAD_14_classpara_Vit_hyperin_s_emaold_u1000_batchmore_zv5_2_OursSGD

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python ./tools/train_val.py --config /dataset/ruizhengwu/SmallDefect/Final_UniAD_14_classpara_Vit_hyperin_s_emaold_u1000_batchmore_zv5_2_OursSGD/experiments/MVTec-AD/config_c1.yaml

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python ./tools/train_val.py --config /dataset/ruizhengwu/SmallDefect/Final_UniAD_14_classpara_Vit_hyperin_s_emaold_u1000_batchmore_zv5_2_OursSGD/experiments/MVTec-AD/config_c15.yaml
