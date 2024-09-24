# ./schedctl create --name our8d --image "harbor.smoa.cc/public/jiaqitang:HDR-v3" --gpu 4 --cmd "/dataset/ruizhengwu/SmallDefect_Vis/Final_UniAD_8d_Ours_final/run.sh" --arch ampere

# export PYTHONPATH=../../:$PYTHONPATH
# pip install scikit-learn
# pip install tensorboardX
# pip install tabulate
# pip install /dataset/ruizhengwu/SmallDefect/torch-1.11.0+cu113-cp38-cp38-linux_x86_64.whl
cd /dataset/ruizhengwu/SmallDefect_Vis/Final_UniAD_8d_Ours_final

CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train_val.py --config /dataset/ruizhengwu/SmallDefect_Vis/Final_UniAD_8d_Ours_final/experiments/MVTec-AD/config_c1.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train_val.py --config /dataset/ruizhengwu/SmallDefect_Vis/Final_UniAD_8d_Ours_final/experiments/MVTec-AD/config_c12.yaml
