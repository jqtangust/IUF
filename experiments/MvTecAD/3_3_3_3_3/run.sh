# ./schedctl create --name our33333 --image "harbor.smoa.cc/public/jiaqitang:HDR-v3" --gpu 4 --cmd "/dataset/ruizhengwu/SmallDefect/Final_UniAD_33333_classpara_Vit_hyperin_s_emaold_u1000_batchmore_zv5_2_OursSGD_v3/run.sh" --arch ampere

# export PYTHONPATH=../../:$PYTHONPATH
# pip install scikit-learn
# pip install tensorboardX
# pip install tabulate
# pip install /dataset/ruizhengwu/SmallDefect/torch-1.11.0+cu113-cp38-cp38-linux_x86_64.whl
cd /dataset/ruizhengwu/SmallDefect/Final_UniAD_33333_classpara_Vit_hyperin_s_emaold_u1000_batchmore_zv5_2_OursSGD_v3

CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train_val.py --config /dataset/ruizhengwu/SmallDefect/Final_UniAD_33333_classpara_Vit_hyperin_s_emaold_u1000_batchmore_zv5_2_OursSGD_v3/experiments/MVTec-AD/config_c3.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train_val.py --config /dataset/ruizhengwu/SmallDefect/Final_UniAD_33333_classpara_Vit_hyperin_s_emaold_u1000_batchmore_zv5_2_OursSGD_v3/experiments/MVTec-AD/config_c6.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train_val.py --config /dataset/ruizhengwu/SmallDefect/Final_UniAD_33333_classpara_Vit_hyperin_s_emaold_u1000_batchmore_zv5_2_OursSGD_v3/experiments/MVTec-AD/config_c9.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train_val.py --config /dataset/ruizhengwu/SmallDefect/Final_UniAD_33333_classpara_Vit_hyperin_s_emaold_u1000_batchmore_zv5_2_OursSGD_v3/experiments/MVTec-AD/config_c12.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train_val.py --config /dataset/ruizhengwu/SmallDefect/Final_UniAD_33333_classpara_Vit_hyperin_s_emaold_u1000_batchmore_zv5_2_OursSGD_v3/experiments/MVTec-AD/config_c15.yaml
