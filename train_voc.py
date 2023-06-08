from cgi import parse_multipart
import os
import logging
import time
from collections import OrderedDict, Counter
import copy 

import numpy as np

import torch
from torch import autograd
import torch.utils.data as torchdata

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.layers.batch_norm import FrozenBatchNorm2d
from detectron2.engine import DefaultPredictor, DefaultTrainer, default_setup
from detectron2.engine import default_argument_parser, hooks, HookBase
from detectron2.solver.build import get_default_optimizer_params, maybe_add_gradient_clipping, build_lr_scheduler
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_train_loader, build_detection_test_loader, get_detection_dataset_dicts
from detectron2.data.common import  DatasetFromList, MapDataset
from detectron2.data.samplers import InferenceSampler
from detectron2.utils.events import  get_event_storage

from detectron2.utils import comm
from detectron2.evaluation import COCOEvaluator, verify_results, inference_on_dataset, print_csv_format

from detectron2.solver import LRMultiplier
from detectron2.modeling import build_model
from detectron2.structures import ImageList, Instances, pairwise_iou, Boxes

from fvcore.common.param_scheduler import ParamScheduler
from fvcore.common.checkpoint import Checkpointer

from data.datasets import builtin

from detectron2.evaluation import  PascalVOCDetectionEvaluator, COCOEvaluator, inference_on_dataset

from detectron2.data import build_detection_train_loader, MetadataCatalog
import torch.utils.data as data
from detectron2.data.dataset_mapper import DatasetMapper
import detectron2.data.detection_utils as utils
import detectron2.data.transforms as detT

import torchvision.transforms as T
import torchvision.transforms.functional as tF

from modeling import add_stn_config
from modeling import CustomPascalVOCDetectionEvaluator

logger = logging.getLogger("detectron2")

def setup(args):
    cfg = get_cfg()
    add_stn_config(cfg)
    #hack to add base yaml 
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_file(model_zoo.get_config_file(cfg.BASE_YAML))
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    #cfg.freeze()
    default_setup(cfg, args)
    return cfg

class CustomDatasetMapper(DatasetMapper):
    def __init__(self,cfg,is_train) -> None:
        super().__init__(cfg,is_train)
        self.with_crops = cfg.INPUT.CLIP_WITH_IMG
        self.with_random_clip_crops = cfg.INPUT.CLIP_RANDOM_CROPS
        self.with_jitter = cfg.INPUT.IMAGE_JITTER
        self.cropfn = T.RandomCrop#T.RandomCrop([224,224])
        self.aug = T.ColorJitter(brightness=.5, hue=.3)
        self.crop_size = cfg.INPUT.RANDOM_CROP_SIZE

    def __call__(self,dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)
        
        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None

        aug_input = detT.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )
       
        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)

        if self.with_jitter:
            dataset_dict["jitter_image"] = self.aug(dataset_dict["image"])
            
        if self.with_crops:
            bbox = dataset_dict['instances'].gt_boxes.tensor
            csx = (bbox[:,0] + bbox[:,2])*0.5
            csy = (bbox[:,1] + bbox[:,3])*0.5
            maxwh = torch.maximum(bbox[:,2]-bbox[:,0],bbox[:,3]-bbox[:,1])
            crops = list()
            gt_boxes = list()
            mean=[0.48145466, 0.4578275, 0.40821073]
            std=[0.26862954, 0.26130258, 0.27577711]    
            for cx,cy,maxdim,label,box in zip(csx,csy,maxwh,dataset_dict['instances'].gt_classes, bbox):

                if int(maxdim) < 10:
                    continue
                x0 = torch.maximum(cx-maxdim*0.5,torch.tensor(0))
                y0 = torch.maximum(cy-maxdim*0.5,torch.tensor(0))
                try:
                    imcrop = T.functional.resized_crop(dataset_dict['image'],top=int(y0),left=int(x0),height=int(maxdim),width=int(maxdim),size=224)
                    imcrop = imcrop.flip(0)/255 # bgr --> rgb for clip
                    imcrop = T.functional.normalize(imcrop,mean,std)
                    # print(x0,y0,x0+maxdim,y0+maxdim,dataset_dict['image'].shape)
                    # print(imcrop.min(),imcrop.max() )
                    gt_boxes.append(box.reshape(1,-1))
                except Exception as e:
                    print(e)
                    print('crops:',x0,y0,maxdim)
                    exit()
                # crops.append((imcrop,label))
                crops.append(imcrop.unsqueeze(0))
            
            if len(crops) == 0:
                dataset_dict['crops'] = []
            else:
                dataset_dict['crops'] = [torch.cat(crops,0),Boxes(torch.cat(gt_boxes,0))]

        if self.with_random_clip_crops:
            crops = []
            rbboxs = []
            
            for i in range(15):
                minsize = min(dataset_dict['image'].shape[1],dataset_dict['image'].shape[2])
                p = self.cropfn.get_params(dataset_dict['image'],[min(self.crop_size,minsize),min(self.crop_size,minsize)])
                c = tF.crop(dataset_dict['image'],*p)
                if self.crop_size != 224:
                    c = tF.resize(img=c,size=224)
                crops.append(c)
                rbboxs.append(p)
            
            crops = torch.stack(crops)
            dataset_dict['randomcrops'] = crops

            #apply same crop bbox to the jittered image
            if self.with_jitter:
                jitter_crops = []
                for p in rbboxs:
                    jc = tF.crop(dataset_dict['jitter_image'],*p) 
                    if self.crop_size != 224:
                        jc = tF.resize(img=jc,size=224)
                    jitter_crops.append(jc)
           
                jcrops = torch.stack(jitter_crops)
                dataset_dict['jitter_randomcrops'] = jcrops

        return dataset_dict

class CombineLoaders(data.IterableDataset):
    def __init__(self,loaders):
        self.loaders = loaders

    def __iter__(self,):
        dd = iter(self.loaders[1])
        for d1 in self.loaders[0]:
            try:
                d2 = next(dd)
            except:
                dd=iter(self.loaders[1])
                d2 = next(dd)

            list_out_dict=[]
            for v1,v2 in zip(d1,d2):
                out_dict = {}
                for k in v1.keys():
                    out_dict[k] = (v1[k],v2[k])
                list_out_dict.append(out_dict)

            yield list_out_dict

    
class Trainer(DefaultTrainer):

    def __init__(self,cfg) -> None:
        super().__init__(cfg)
        self.teach_model = None
        self.off_opt_interval = np.arange(0,cfg.SOLVER.MAX_ITER,cfg.OFFSET_OPT_INTERVAL[0]).tolist()
        self.off_opt_iters = cfg.OFFSET_OPT_ITERS

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:
        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = build_model(cfg)
        

        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))

        return model

    @classmethod
    def build_train_loader(cls,cfg):
        original  = cfg.DATASETS.TRAIN
        print(original)
        # cfg.DATASETS.TRAIN=(original[0],)
        data_loader1 = build_detection_train_loader(cfg, mapper=CustomDatasetMapper(cfg, True))
        return data_loader1

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        if MetadataCatalog.get(dataset_name).evaluator_type == 'pascal_voc':
            return CustomPascalVOCDetectionEvaluator(dataset_name)
        else:
            return COCOEvaluator(dataset_name, output_dir=output_folder)

    @classmethod
    def build_optimizer(cls,cfg,model):
        
        trainable = {'others':[],'offset':[]}

        for name,val in model.named_parameters():
            head = name.split('.')[0]
            #previously was setting all params to be true
            if val.requires_grad == True:
                print(name)
                if 'offset' in name:
                    trainable['offset'].append(val)   
                else:
                    trainable['others'].append(val)

        optimizer1 = torch.optim.SGD(
            trainable['others'],
            cfg.SOLVER.BASE_LR,
            momentum=cfg.SOLVER.MOMENTUM,
            nesterov=cfg.SOLVER.NESTEROV,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )

        optimizer2 = torch.optim.Adam(
            trainable['offset'],
            0.01,
        )
        return (maybe_add_gradient_clipping(cfg, optimizer1),maybe_add_gradient_clipping(cfg, optimizer2))

    
    def run_step(self):
        """
        Implement the standard training logic described above.
        """


        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._trainer._data_loader_iter)
        data_time = time.perf_counter() - start

        """
        If you want to do something with the losses, you can wrap the model.
        """
        data_s = data

        opt_phase = False
        if len(self.off_opt_interval) and self.iter >= self.off_opt_interval[0] and self.iter < self.off_opt_interval[0]+self.off_opt_iters:
        
            if self.iter == self.off_opt_interval[0]:
                self.model.offsets.data = torch.zeros(self.model.offsets.shape).cuda()
            loss_dict_s = self.model.opt_offsets(data_s)
            opt_phase = True
            if self.iter+1 == self.off_opt_interval[0]+self.off_opt_iters:
                self.off_opt_interval.pop(0)
                
        else:  
            # for ind, d in enumerate(data_s):
            #     d['image'] = self.aug(d['image'].cuda())
            loss_dict_s = self.model(data_s)
            # print(loss_dict_s)
        
        # import pdb;pdb.set_trace()
        loss_dict = {}

        loss = 0 
        for k,v in loss_dict_s.items():
            loss += v

        
        """
        If you need to accumulate gradients or do something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer[0].zero_grad() 
        self.optimizer[1].zero_grad()

        loss.backward()   

        if not opt_phase:
            self.optimizer[0].step() 
        else:
            self.optimizer[1].step()

        self.optimizer[0].zero_grad()
        self.optimizer[1].zero_grad()
        
        for k,v in loss_dict_s.items():
            loss_dict.update({k:v})
        
        # print(loss_di ct)
        self._trainer._write_metrics(loss_dict, data_time)
        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method. But it is
        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
        """

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.
        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            LRScheduler(),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        def do_test_st(flag):
            if flag == 'st':
                model = self.model 
            else:
                print("Error in the flag")

            results = OrderedDict()
            for dataset_name in self.cfg.DATASETS.TEST:
                data_loader = build_detection_test_loader(self.cfg, dataset_name)
                evaluator = CustomPascalVOCDetectionEvaluator(dataset_name)
                results_i = inference_on_dataset(model, data_loader, evaluator)
                results[dataset_name] = results_i
                if comm.is_main_process():
                    logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                    print_csv_format(results_i)
                    storage = get_event_storage()
                    storage.put_scalar(f'{dataset_name}_AP50', results_i['bbox']['AP50'],smoothing_hint=False)
            if len(results) == 1:
                results = list(results.values())[0]
            return results
        
       
        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_SAVE_PERIOD, lambda flag='st': do_test_st(flag)))

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """

        return build_lr_scheduler(cfg, optimizer[0])

    def state_dict(self):
        ret = super().state_dict()
        ret["optimizer1"] = self.optimizer[0].state_dict()
        ret["optimizer2"] = self.optimizer[1].state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.optimizer[0].load_state_dict(state_dict["optimizer1"])
        self.optimizer[1].load_state_dict(state_dict["optimizer2"])



class LRScheduler(HookBase):
    """
    A hook which executes a torch builtin LR scheduler and summarizes the LR.
    It is executed after every iteration.
    """

    def __init__(self, optimizer=None, scheduler=None):
        """
        Args:
            optimizer (torch.optim.Optimizer):
            scheduler (torch.optim.LRScheduler or fvcore.common.param_scheduler.ParamScheduler):
                if a :class:`ParamScheduler` object, it defines the multiplier over the base LR
                in the optimizer.
        If any argument is not given, will try to obtain it from the trainer.
        """
        self._optimizer = optimizer
        self._scheduler = scheduler

    def before_train(self):
        self._optimizer = self._optimizer or self.trainer.optimizer
        if isinstance(self.scheduler, ParamScheduler):
            self._scheduler = LRMultiplier(
                self._optimizer,
                self.scheduler,
                self.trainer.max_iter,
                last_iter=self.trainer.iter - 1,
            )
        self._best_param_group_id1 = LRScheduler.get_best_param_group_id(self._optimizer[0])
        self._best_param_group_id2 = LRScheduler.get_best_param_group_id(self._optimizer[1])


    @staticmethod
    def get_best_param_group_id(optimizer):
        # NOTE: some heuristics on what LR to summarize
        # summarize the param group with most parameters
        largest_group = max(len(g["params"]) for g in optimizer.param_groups)

        if largest_group == 1:
            # If all groups have one parameter,
            # then find the most common initial LR, and use it for summary
            lr_count = Counter([g["lr"] for g in optimizer.param_groups])
            lr = lr_count.most_common()[0][0]
            for i, g in enumerate(optimizer.param_groups):
                if g["lr"] == lr:
                    return i
        else:
            for i, g in enumerate(optimizer.param_groups):
                if len(g["params"]) == largest_group:
                    return i

    def after_step(self):
        lr1 = self._optimizer[0].param_groups[self._best_param_group_id1]["lr"]
        self.trainer.storage.put_scalar("lr1", lr1, smoothing_hint=False)

        lr2 = self._optimizer[1].param_groups[self._best_param_group_id2]["lr"]
        self.trainer.storage.put_scalar("lr2", lr2, smoothing_hint=False)

        self.scheduler.step()

    @property
    def scheduler(self):
        return self._scheduler or self.trainer.scheduler

    def state_dict(self):
        if isinstance(self.scheduler, torch.optim.lr_scheduler._LRScheduler):
            return self.scheduler.state_dict()
        return {}

    def load_state_dict(self, state_dict):
        if isinstance(self.scheduler, torch.optim.lr_scheduler._LRScheduler):
            logger = logging.getLogger(__name__)
            logger.info("Loading scheduler from state_dict ...")
            self.scheduler.load_state_dict(state_dict)

def custom_build_detection_test_loader(cfg,dataset_name,mapper=None):

    if isinstance(dataset_name, str):
        dataset_name = [dataset_name]

    dataset = get_detection_dataset_dicts(
        dataset_name,
        filter_empty=False,
        proposal_files=[
            cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(x)] for x in dataset_name
        ]
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )
    if mapper is None:
        mapper = DatasetMapper(cfg, False)
   
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)

    sampler = None
    if isinstance(dataset, torchdata.IterableDataset):
        assert sampler is None, "sampler must be None if dataset is IterableDataset"
    else:
        if sampler is None:
            sampler = InferenceSampler(len(dataset))
    collate_fn  = None

    def trivial_batch_collator(batch):
        """
        A batch collator that does nothing.
        """
        return batch

    return torchdata.DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        drop_last=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        collate_fn=trivial_batch_collator if collate_fn is None else collate_fn,
    )


def do_test(cfg, model, model_type=''):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)#custom_build_detection_test_loader(cfg, dataset_name,CustomDatasetMapper(cfg,is_train=True))
        evaluator = CustomPascalVOCDetectionEvaluator(dataset_name)#COCOEvaluator(dataset_name, output_dir=os.path.join(cfg.OUTPUT_DIR, "inference"))
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)

        if len(results) == 1:       
            results = list(results.values())[0]
    return results

def main(args):
    cfg = setup(args)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(cfg,model)
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    for dataset_name in cfg.DATASETS.TEST:
        if '_val' in dataset_name :
            trainer.register_hooks([

                    hooks.BestCheckpointer(cfg.TEST.EVAL_SAVE_PERIOD,trainer.checkpointer,f'{dataset_name}_AP50',file_prefix='model_best'),
                    ])

    trainer.train()
    

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    cfg = setup(args)
    print("Command Line Args:", args)

    main(args)
