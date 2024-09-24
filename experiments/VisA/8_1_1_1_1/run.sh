# ./schedctl create --name our8 --image "harbor.smoa.cc/public/jiaqitang:HDR-v3" --gpu 4 --cmd "/dataset/ruizhengwu/SmallDefect_Vis/IUF/experiments/VisA/8_1_1_1_1/run.sh" --arch ampere

# export PYTHONPATH=../../:$PYTHONPATH
# pip install scikit-learn
# pip install tensorboardX
# pip install tabulate
# pip install /dataset/ruizhengwu/SmallDefect/torch-1.11.0+cu113-cp38-cp38-linux_x86_64.whl
cd /dataset/ruizhengwu/SmallDefect_Vis/IUF

# Stage 1: Training base model
# CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train_val.py --config /dataset/ruizhengwu/SmallDefect_Vis/IUF/experiments/VisA/8_1_1_1_1/config_c1.yaml

#Stage 2: Training incremental model
CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train_val.py --config /dataset/ruizhengwu/SmallDefect_Vis/IUF/experiments/VisA/8_1_1_1_1/config_c9.yaml

# CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train_val.py --config /dataset/ruizhengwu/SmallDefect_Vis/IUF/experiments/VisA/8_1_1_1_1/config_c10.yaml

# CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train_val.py --config /dataset/ruizhengwu/SmallDefect_Vis/IUF/experiments/VisA/8_1_1_1_1/config_c11.yaml

# CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train_val.py --config /dataset/ruizhengwu/SmallDefect_Vis/IUF/experiments/VisA/8_1_1_1_1/config_c12.yaml
