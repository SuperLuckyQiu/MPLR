import torch
import torch.nn as nn
from torch.nn import functional as F

from .clip_text import CLIP_Text
from .clip_text import CLIP_Coop
from .peft_vit import Peft_ViT, ViT_Tuner
from .peft_rn import Peft_RN, RN_Tuner
from .classifiers import *
from .clip_text import PromptLearner
from clip import clip

class ZeroShotCLIP(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.text_encoder = CLIP_Text(clip_model)
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale.exp()
        self.dtype = clip_model.dtype

    def encode_text(self, text):
        try:
            text_features = self.text_encoder(text)
        except:
            # CUDA out of memory
            text_split = torch.split(text, 1000)
            text_features = torch.cat([self.text_encoder(x) for x in text_split])
        return text_features

    def encode_image(self, image):
        return self.image_encoder(image.to(self.dtype))
    
    @torch.no_grad()
    def init_text_features(self, prompts):
        text_features = self.encode_text(prompts)
        text_features = F.normalize(text_features, dim=-1)
        self.text_features = text_features
    
    def forward(self, image):
        image_features = self.encode_image(image)
        image_features = F.normalize(image_features, dim=-1)
        logit = self.logit_scale * F.linear(image_features, self.text_features)
        return logit


class PeftModelFromCLIP(nn.Module):
    def __init__(self, cfg, clip_model, num_classes, classnames):
        super().__init__()

        if cfg.backbone.startswith("CLIP-ViT"):
            self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
            self.tokenized_prompts = self.prompt_learner.tokenized_prompts
            self.text_encoder_coop = CLIP_Coop(clip_model)
            self.prompt_text_features= None        
            self.image_encoder = Peft_ViT(clip_model.visual,clip_model)
            self.tuner = ViT_Tuner(cfg, clip_model.visual, num_classes,clip_model)
            self.proj = clip_model.visual.proj
            self.logit_scale = clip_model.logit_scale
            self.dtype = clip_model.dtype
            self.a = torch.nn.Parameter(torch.ones(1, num_classes))
            self.b = torch.nn.Parameter(torch.zeros(1, num_classes))
            self.trainable_proj_768 = torch.nn.Parameter(clip_model.visual.proj)
            self.trainable_proj_512 = torch.nn.Parameter(clip_model.visual.proj.t())
        
        feat_dim = self.image_encoder.out_dim
        dtype = self.image_encoder.dtype
        self.head = eval(cfg.classifier)(feat_dim, num_classes, dtype, **cfg)
        self.w_norm = torch.norm(self.head.weight.data,dim=1).to("cuda")
        self.other_parameters = nn.ParameterList([self.a, self.b,])
    @torch.no_grad()        
    def encode_text(self, text):
        try:
            text_features = self.text_encoder_coop(text)
        except:
            # CUDA out of memory
            text_split = torch.split(text, 1000)
            text_features = torch.cat([self.text_encoder_coop(x,None,False) for x in text_split])
        return text_features
    def encode_label(self,labels):
        template = " a photo of a {}."
        labels = [str(t.tolist()) for t in labels]
        prompts = [template.format(c.replace("_", " ")) for c in labels]
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to("cuda")
        text_features= self.encode_text(prompts)
        text_features = text_features @ self.proj.t()
        text_features = F.normalize(text_features, dim=-1)
        return text_features
    def coop(self):
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder_coop(prompts, tokenized_prompts,True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features   


    def forward(self, image,labels = None, use_tuner=True, return_feature=False):
        # coop
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts 
        text_features = self.text_encoder_coop(prompts, tokenized_prompts,True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)  
        if labels is not  None:
            labels_features = self.encode_label(labels)
        logit_scale = self.logit_scale.exp()
        tuner = self.tuner if use_tuner else None
        head = self.head if not return_feature else None
        with torch.no_grad():
            zero_image_features=self.image_encoder.encode_image(image.type(self.dtype))
            zero_image_features = zero_image_features / zero_image_features.norm(dim=-1, keepdim=True)   
            prompt_text_features=self.prompt_text_features
        
        logits_prompt = logit_scale * zero_image_features @ prompt_text_features.t()
        _, indices = torch.sort(logits_prompt, descending=True)
        indices = indices[:, :1]
        text = torch.cat((text_features[indices],prompt_text_features[indices]),dim=1) 
        if labels is not  None:
            vpt_image_features = self.image_encoder(image, text,labels_features,tuner, head)
        else:
            vpt_image_features = self.image_encoder(image, text,tuner=tuner,head=head)
        ratio = 0.1
        vpt_image_features=(1-ratio)*vpt_image_features+ratio*(zero_image_features @ self.trainable_proj_512)
        logits = head(vpt_image_features)
        vpt_image_features = vpt_image_features@ self.trainable_proj_768
        vpt_image_features = vpt_image_features / vpt_image_features.norm(dim=-1, keepdim=True)
        logits1 = logit_scale * vpt_image_features  @ text_features.t()
        logits1 =self.a *logits1 + self.b * self.w_norm
        return [logits,logits1,]

