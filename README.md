# understand-ssl-part-aware
Official implementation for TMLR Paper: "Understanding Self-Supervised Pretraining with Part-Aware Representation Learning" [[Arxiv](https://arxiv.org/abs/2301.11915)] [[TMLR](https://openreview.net/pdf?id=HP7Qpui5YE)]

## Abstract
In this paper, we are interested in understanding self-supervised pretraining through studying
the capability that self-supervised methods learn part-aware representations. The study is
mainly motivated by that random views, used in contrastive learning, and random masked
(visible) patches, used in masked image modeling, are often about object parts.

We explain that contrastive learning is a part-to-whole task: the projection layer hallucinates
the whole object representation from the object part representation learned from the encoder,
and that masked image modeling is a part-to-part task: the masked patches of the object
are hallucinated from the visible patches. The explanation suggests that the self-supervised
pretrained encoder leans toward understanding the object part. We empirically compare
the off-the-shelf encoders pretrained with several representative methods on object-level
recognition and part-level recognition. The results show that the fully-supervised model
outperforms self-supervised models for object-level recognition, and most self-supervised
contrastive learning and masked image modeling methods outperform the fully-supervised
method for part-level recognition. It is observed that the combination of contrastive learning
and masked image modeling further improves the performance.

## Code
This code contains three types of part-level tasks including part retrieval, part classification, and part segmentation. 

## Models and Datasets

For all the models involved in the experiments including DeiT, MoCo v3, DINO, BEiT, MAE, CAE, and iBOT, we use their official code to implement the encoders. Note that
for DINO and iBOT, we choose the checkpoint of the teacher models as they have been reported to perform
better than the student models in their papers.

The part and object datasets including ADE20K Part and Object, Pascal Part and Object, and LIP Part are avaliable at [[Google Drive](https://drive.google.com/drive/folders/1JSNzbxc9MBpNIhMRP8Vd0FFk7K5hC65q?usp=sharing)]


## Reference 

if it is helpful, please cite our paper:
```python
@article{zhu2023understanding,
  title={Understanding Self-Supervised Pretraining with Part-Aware Representation Learning},
  author={Zhu, Jie and Qi, Jiyang and Ding, Mingyu and Chen, Xiaokang and Luo, Ping and Wang, Xinggang and Liu, Wenyu and Wang, Leye and Wang, Jingdong},
  journal={arXiv preprint arXiv:2301.11915},
  year={2023}
}
```





