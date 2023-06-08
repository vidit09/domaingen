# CLIP the Gap: A Single Domain Generalization Approach for Object Detection

[ [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Vidit_CLIP_the_Gap_A_Single_Domain_Generalization_Approach_for_Object_CVPR_2023_paper.pdf) ]

### Installation
Our code is based on [Detectron2](https://github.com/facebookresearch/detectron2) and requires python >= 3.6

Install the required packages
```
pip install -r requirements.txt
```

### Datasets
Set the environment variable DETECTRON2_DATASETS to the parent folder of the datasets

```
    path-to-parent-dir/
        /diverseWeather
            /daytime_clear
            /daytime_foggy
            ...
        /comic
        /watercolor
        /VOC2007
        /VOC2012 

```
Download [Diverse Weather](https://github.com/AmingWu/Single-DGOD) and [Cross-Domain](https://naoto0804.github.io/cross_domain_detection/) Datasets and place in the structure as shown.

### Training
We train our models on a single A100 GPU.
```
    python train.py --config-file configs/diverse_weather.yaml 

    or 

    python train_voc.py --config-file configs/comic_watercolor.yaml
```

### Weights
[Download](https://drive.google.com/file/d/1qMJfMZkE7cG6wwphQtA4uAxfh0NBVItu/view?usp=drive_link) the trained weights.

### Citation
```bibtex
@InProceedings{Vidit_2023_CVPR,
    author    = {Vidit, Vidit and Engilberge, Martin and Salzmann, Mathieu},
    title     = {CLIP the Gap: A Single Domain Generalization Approach for Object Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {3219-3229}
}

```
