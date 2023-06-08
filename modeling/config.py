from detectron2.config import  CfgNode as CN

def add_stn_config(cfg):
    cfg.OFFSET_DOMAIN = ''
    cfg.OFFSET_FROZENBN = False
    cfg.OFFSET_DOMAIN_TEXT = ''
    cfg.OFFSET_NAME = 0  
    cfg.OFFSET_OPT_INTERVAL = [10]
    cfg.OFFSET_OPT_ITERS = 0
    cfg.AUG_PROB = 0.5
    cfg.DOMAIN_NAME = ""
    cfg.TEST.EVAL_SAVE_PERIOD  = 5000
    cfg.INPUT.CLIP_WITH_IMG = False
    cfg.INPUT.CLIP_RANDOM_CROPS = False
    cfg.INPUT.IMAGE_JITTER = False
    cfg.INPUT.RANDOM_CROP_SIZE = 224
    cfg.MODEL.GLOBAL_GND = False
    cfg.BASE_YAML = "COCO-Detection/faster_rcnn_R_50_C4_3x.yaml"
    cfg.MODEL.RENAME = list()
    cfg.MODEL.CLIP_IMAGE_ENCODER_NAME = 'ViT-B/32'
    cfg.MODEL.BACKBONE.UNFREEZE  = ['layer3','layer4','attnpool']
    cfg.MODEL.USE_PROJ = True
    