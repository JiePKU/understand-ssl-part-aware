"""
Here is some command example for each task
"""

""" 
for pr_patch_retrieval_imagenet_vis
"""

python3 pr_patch_retrieval.py --pretrained_model cae \
  -a vit_base -j 16 --gpu 5 /home/ssd9/wangxiaodi03/workspace/wangxiaodi03/data/imagenet/val/

"""
for pr_patch_retrieval
"""

## for cub 

python3 pr_patch_retrieval_cub200.py --pretrained_model cae \
  -a vit_base -j 16 --gpu 1 /home/ssd8/hanshumin/imagenet1k/val/

## for coco

python3 pr_patch_retrieval_cub200.py --pretrained_model cae --dataset_name coco \
  -a vit_base -j 16 --gpu 0 /home/ssd8/hanshumin/imagenet1k/val/


"""
for patch_lincls
"""

### attentive

# coco 
mkdir -p logs/cae/
python3 part_lincls.py \
    -a vit_base --lr 0.01 --dist-url 'tcp://localhost:8948' --multiprocessing-distributed --world-size 1 --rank 0 \
     --pretrained_model cae --model_scale base --dataset_name coco --cls_head_type attentive &>> logs/cae/train_attn.log

# cub 
mkdir -p logs/mae_base_1600ep_blockmask0.5/
python3 part_lincls.py \
    -a vit_base --lr 0.01 --dist-url 'tcp://localhost:8947' --multiprocessing-distributed --world-size 1 --rank 0 \
     --pretrained_model cae --model_scale mae_base_1600ep_blockmask0.5 --cls_head_type attentive &>> logs/mae_base_1600ep_blockmask0.5/train_cub200_attn.log


### linear 

# coco

mkdir -p logs/mae_base_300ep/
python3 part_lincls.py \
    -a vit_base --lr 0.1 --dist-url 'tcp://localhost:8947' --multiprocessing-distributed --world-size 1 --rank 0 \
     --pretrained_model cae --model_scale mae_base_300ep --dataset_name coco &>> logs/mae_base_300ep/train.log


# cub

mkdir -p logs/cae/
python3 part_lincls.py \
    -a vit_base --lr 0.1 --dist-url 'tcp://localhost:8947' --multiprocessing-distributed --world-size 1 --rank 0 \
     --pretrained_model cae --model_scale base &>> logs/cae/train_cub200.log



