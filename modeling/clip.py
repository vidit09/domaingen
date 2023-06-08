import clip

import torch
import torch.nn as nn
import time
import numpy as np
import copy

class ClipPredictor(nn.Module):
    def __init__(self, clip_enocder_name,inshape, device, clsnames):
        super().__init__()
        self.model, self.preprocess = clip.load(clip_enocder_name, device)
        self.model.float()
        #freeze everything
        for name, val in self.model.named_parameters():
            val.requires_grad = False
        # this is only used for inference   
        self.frozen_clip_model = copy.deepcopy(self.model)

        self.visual_enc = self.model.visual
        prompt = 'a photo of a {}'
        print(clsnames)
        with torch.no_grad():
            text_inputs = torch.cat([clip.tokenize(prompt.format(cls)) for cls in clsnames]).to(device)
            self.text_features = self.model.encode_text(text_inputs).float()
        self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
        

        self.projection = nn.Linear(inshape,512)
        self.projection_global = nn.Linear(inshape,512)
    
   
    
    def forward(self, feat, gfeat=None):

        if feat.shape[-1] > 512:
            feat = self.projection(feat)
        
        feat = feat/feat.norm(dim=-1,keepdim=True)
        if gfeat is not None:
            
            feat = feat-gfeat
            feat = feat/feat.norm(dim=-1,keepdim=True)
        scores =  (100.0 * torch.matmul(feat,self.text_features.detach().T))

        # print(scores.min(),scores.max())
        # add for bkg class a score 0
        scores = torch.cat([scores,torch.zeros(scores.shape[0],1,device=scores.device)],1) 
        return scores
                                            
    
    


    


   

    
