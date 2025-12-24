# SPRMamba: Surgical Phase Recognition for Endoscopic Submucosal Dissection with Mamba
This repository contains the software used in our Paper “SPRMamba: Surgical Phase Recognition for Endoscopic Submucosal Dissection with Mamba ”
# Enviroment
causal-conv1d == 1.4.0, mamba-ssm ==2.2.2, torch == 2.1.2+cu118, python == 3.10.18, cudatoolkit == 11.8.0, cudnn == 9.10.1.4
# Data Preparation
-We use the dataset [ESD385](https://figshare.com/s/783242188a8a2721a58e)(https://figshare.com/articles/dataset/Renji_endoscopic_submucosal_dissection_video_data_set_for_colorectal_neoplastic_lesions/28737686?file=56338025)(https://figshare.com/articles/dataset/Renji_endoscopic_submucosal_dissection_video_data_set_for_early_gastric_cancer/28045295) and [Cholec80](http://camma.u-strasbg.fr/datasets/). You can download ESD385 partly
-Training and test data split
ESD385: run split.py to generate train.csv, val.csv, test.csv respectively。
Cholec80: first 40 videos for training and the rest 40 videos for testing, following the original paper [EndoNet](https://arxiv.org/abs/1602.03012).
-Data Preprocessing
run preprocessing.py to convert the videos to frames
# Train your own model
- run train.py to train ResNet50
- run generate_LFB.py to generate spatial embeddings
- run main.py to train SPRMamba
# Cite
@article{zhang2024sprmamba,
  title={SPRMamba: Surgical Phase Recognition for Endoscopic Submucosal Dissection with Mamba},
  author={Zhang, Xiangning and Zhang, Qingwei and Chen, Jinnan and Zhou, Chengfeng and Wang, Yaqi and Zhang, Zhengjie and Li, Xiaobo and Qian, Dahong},
  journal={arXiv preprint arXiv:2409.12108},
  year={2024}
}
# License
This project is released under the [MIT License](https://github.com/Zxnyyyyy/SPRMamba/blob/main/LICENSE)
# Acknowledgement
This repository is built based on [ASFormer](https://github.com/ChinaYi/ASFormer) repository.
