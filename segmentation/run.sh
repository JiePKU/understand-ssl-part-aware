


# mae for lip lp freeze12
cd /root/paddlejob/workspace/env_run/cae_partseg/CAE-master/downstream_tasks/semantic_segmentation
mkdir -p logs/mae_for_lip_partseg_lp_freeze12/
bash tools/dist_train.sh \
    configs_local/mae/upernet/upernet_mae_base_12_320_slide_160k_lip_pt_4e-4_lp_freeze12.py 8 \
    --work-dir logs/mae_for_lip_partseg_lp_freeze12/ --seed 0 --deterministic \
    --options model.pretrained=pretrained/mae_pretrain_vit_base.pth &>> logs/mae_for_lip_partseg_lp_freeze12/train.log 


sleep 20s
# moco for lip lp freeze12
cd /root/paddlejob/workspace/env_run/cae_partseg/CAE-master/downstream_tasks/semantic_segmentation
mkdir -p logs/moco_for_lip_partseg_lp_freeze12/
bash tools/dist_train.sh \
    configs_local/moco/upernet/upernet_moco_base_12_320_slide_160k_lip_pt_4e-4_lp_freeze12.py 8 \
    --work-dir logs/moco_for_lip_partseg_lp_freeze12/ --seed 0 --deterministic \
    --options model.pretrained=pretrained/moco-vit-b-300ep.pth &>> logs/moco_for_lip_partseg_lp_freeze12/train.log 


# deit for lip lp freeze12
cd /root/paddlejob/workspace/env_run/cae_partseg/CAE-master/downstream_tasks/semantic_segmentation
mkdir -p logs/deit_for_lip_partseg_lp_freeze12/
bash tools/dist_train.sh \
    configs_local/deit/upernet/upernet_deit_base_12_320_slide_160k_lip_pt_4e-4_lp_freeze12.py 8 \
    --work-dir logs/deit_for_lip_partseg_lp_freeze12/ --seed 0 --deterministic \
    --options model.pretrained=pretrained/deit_base_distilled_patch16_224-df68dfff.pth &>> logs/deit_for_lip_partseg_lp_freeze12/train.log 
