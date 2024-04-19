import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from clip.model import VisionTransformer as CLIP_ViT
from timm.models.vision_transformer import VisionTransformer as ViT

from .peft_modules import *


class ViT_Tuner(nn.Module):
    """ All instance variables in this class will be optimized.
    """
    def __init__(self, cfg, vit_model, num_classes,clip_model):
        super().__init__()

        if isinstance(vit_model, CLIP_ViT):
            n_layers = len(vit_model.transformer.resblocks)
            emb_dim = vit_model.positional_embedding.shape[1]
            seq_len = vit_model.positional_embedding.shape[0]
            patch_size = vit_model.conv1.kernel_size
            dtype = vit_model.conv1.weight.dtype

            blocks = vit_model.transformer.resblocks
            attn_in_dim = blocks[0].attn.in_proj_bias.shape[0]
            attn_out_dim = blocks[0].attn.out_proj.bias.shape[0]
            mlp_in_dim = blocks[0].mlp[0].bias.shape[0]
            mlp_out_dim = blocks[0].mlp[2].bias.shape[0]

        elif isinstance(vit_model, ViT):
            n_layers = len(vit_model.blocks)
            emb_dim = vit_model.pos_embed.shape[2]
            seq_len = vit_model.pos_embed.shape[1]
            patch_size = vit_model.patch_embed.proj.kernel_size
            dtype = vit_model.patch_embed.proj.weight.dtype

            blocks = vit_model.blocks
            attn_in_dim = blocks[0].attn.qkv.bias.shape[0]
            attn_out_dim = blocks[0].attn.proj.bias.shape[0]
            mlp_in_dim = blocks[0].mlp.fc1.bias.shape[0]
            mlp_out_dim = blocks[0].mlp.fc2.bias.shape[0]

        use_vpt_shallow = cfg.vpt_shallow
        use_vpt_deep = cfg.vpt_deep
        partial = cfg.partial
        vpt_len = cfg.vpt_len


        if partial is None:
            partial = n_layers
        
        if (use_vpt_shallow or use_vpt_deep) and (vpt_len is None):
            vpt_len = 10
            print("Visual prompt length set to {}".format(vpt_len))


        assert int(use_vpt_shallow) + int(use_vpt_deep) < 2
        if use_vpt_shallow:
            vpt_list = nn.ModuleList([
                VPT(vpt_len=vpt_len, seq_len=seq_len, patch_size=patch_size, emb_dim=emb_dim, dtype=dtype),
                *[None] * (n_layers - 1)
            ])
        elif use_vpt_deep:
            vpt_list = nn.ModuleList([
                *[None] * (n_layers - partial),
                *[VPT(vpt_len=vpt_len, seq_len=seq_len, patch_size=patch_size, emb_dim=emb_dim, dtype=dtype) for _ in range(partial)]
            ])
        else:
            vpt_list = nn.ModuleList([None] * n_layers)
       
        
        # To be optimized
        self.vpt_list = vpt_list
        
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.norm1q = nn.LayerNorm(dim)
        self.norm1k = nn.LayerNorm(dim)

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)


    def forward(self, qx: torch.Tensor, kx: torch.Tensor, key_padding_mask: torch.Tensor = None):
        # qx: [Bq, 1, C]    kx: [Bk, Nk, C]
        # key_padding_mask: [Bk, Nk] (mask==1 ==> '-inf')
        # output: [Bq, Bk, C]

        assert qx.shape[-1] == kx.shape[-1] 
        Bq, _, C = qx.shape
        Bk, Nk, _ = kx.shape
        q = self.wq(self.norm1q(qx))
        q = q.reshape(Bq, 1, self.num_heads, C //
                                             self.num_heads).permute(0, 2, 1, 3)
        k = self.wk(self.norm1k(kx)).reshape(Bk, Nk, self.num_heads, C //
                                             self.num_heads).permute(0, 2, 1, 3)
        v = kx.unsqueeze(1)
        #  q: [Bq, num_heads,  1, C // num_heads]
        # kv: [Bk, num_heads, Nk, C // num_heads]
        # attn: [Bq, Bk, num_heads, Nk]
        attn = torch.einsum('qhoc,khnc->qkhn ', q, k) * self.scale

        if key_padding_mask is not None:
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(0).unsqueeze(2), float('-inf'),
            )
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = torch.einsum('khnc,qkhn->qkc', v, attn)
        x = x.reshape(Bq,-1 , C)
        return x
        
class Peft_ViT(nn.Module):
    def __init__(self, vit_model,clip_model):
        super().__init__()

        if isinstance(vit_model, CLIP_ViT):
            self.backbone = "CLIP-VIT"
            self.patch_embedding = vit_model.conv1
            self.class_embedding = vit_model.class_embedding
            self.positional_embedding = vit_model.positional_embedding
            self.ln_pre = vit_model.ln_pre
            self.blocks = vit_model.transformer.resblocks
            self.ln_post = vit_model.ln_post
            self.proj = vit_model.proj  # not used
            self.out_dim = self.ln_post.bias.shape[0]
            # self.out_dim = self.proj.shape[1]
            self.meta_nets = nn.Linear(512,768)   
            # self.meta_nets = clip_model.visual.proj.t().to("cuda")     
            self.attn1 = Attention(768, 8, qkv_bias=False, qk_scale=None,attn_drop=0, proj_drop=0)    
            self.clip_model = clip_model    
        elif isinstance(vit_model, ViT):
            self.backbone = "ViT"
            self.patch_embedding = vit_model.patch_embed.proj
            self.class_embedding = vit_model.cls_token
            self.positional_embedding = vit_model.pos_embed
            self.ln_pre = vit_model.norm_pre
            self.blocks = vit_model.blocks
            self.ln_post = vit_model.norm
            self.proj = nn.Identity()
            self.out_dim = self.ln_post.bias.shape[0]
    @torch.no_grad()      
    def encode_image(self,x):
        return self.clip_model.visual(x)

    @property
    def dtype(self):
        return self.patch_embedding.weight.dtype

    def forward(self, x,text_features,labels_features = None, tuner=None, head=None):
        x = x.to(self.dtype)
        x = self.patch_embedding(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        _bsz = x.shape[0]
        _seq_len = x.shape[1]
        _emb_dim = x.shape[2]

        n_layers = len(self.blocks)

        for i in range(n_layers):
            block = self.blocks[i]

            if tuner is not None:
                vpt = tuner.vpt_list[i]
                adapter = tuner.adapter_list[i]
                adaptformer = tuner.adaptformer_list[i]
                lora = tuner.lora_list[i]
                ssf_attn = tuner.ssf_attn_list[i]
                ssf_mlp = tuner.ssf_mlp_list[i]
                ssf_ln = tuner.ssf_ln_list[i]
            else:
                vpt = adapter = adaptformer = lora = ssf_attn = ssf_mlp = ssf_ln = None

            

            if vpt is not None :

                x = vpt(x)
                if i==(n_layers-1): 
  
                    text_feature = self.meta_nets(text_features)
                    v = self.attn1(x[:, 0, :].unsqueeze(1), text_feature.permute(1,0,2))
                    qx_ = F.normalize(x, p=2, dim=-1)
                    v_ = v / 21.1578
                    x2 = torch.einsum('qkc,qoc->qkc', v_, qx_)
                    
                    if labels_features is not None:
                        labels_feature = labels_features.unsqueeze(1).expand(-1,2,-1)
                        x =  torch.cat([x, x2+labels_feature], dim=1)   
                    else:
                        x =  torch.cat([x, x2+1.1*text_feature], dim=1)                

                               

            _seq_len_after_vpt = x.shape[1]

            x = x.permute(1, 0, 2)  # NLD -> LND

            if self.backbone == "CLIP-VIT":
                _attn = block.attn
                _ln_1 = block.ln_1
                _mlp = block.mlp
                _ln_2 = block.ln_2

                _attn_in_proj_weight = _attn.in_proj_weight
                _attn_in_proj_bias = _attn.in_proj_bias
                _attn_out_proj_weight = _attn.out_proj.weight
                _attn_out_proj_bias = _attn.out_proj.bias
                _mlp_in_proj = _mlp[0]
                _mlp_act = _mlp[1]
                _mlp_out_proj = _mlp[2]

                _num_heads = _attn.num_heads
                _head_dim = _emb_dim // _num_heads
            
            elif self.backbone == "ViT":
                _attn = block.attn
                _ln_1 = block.norm1
                _mlp = block.mlp
                _ln_2 = block.norm2

                _attn_in_proj_weight = _attn.qkv.weight
                _attn_in_proj_bias = _attn.qkv.bias
                _attn_out_proj_weight = _attn.proj.weight
                _attn_out_proj_bias = _attn.proj.bias
                _mlp_in_proj = _mlp.fc1
                _mlp_act = _mlp.act
                _mlp_out_proj = _mlp.fc2

                _num_heads = _attn.num_heads
                _head_dim = _emb_dim // _num_heads

            ###############################
            ## Multi-Head Self-Attention ##
            ###############################
            identity = x  # deep copy

            x = _ln_1(x)
            if ssf_ln is not None:
                x = ssf_ln["ln_1"](x)

            qkv = F.linear(x, _attn_in_proj_weight, _attn_in_proj_bias)
            if ssf_attn is not None:
                qkv = ssf_attn["attn_in"](qkv)

            q, k, v = qkv.chunk(3, dim=-1)

            if lora is not None:
                q = q + lora["q"](x)
                v = v + lora["v"](x)

            q = q.contiguous().view(q.shape[0], q.shape[1] * _num_heads, _head_dim).transpose(0, 1)
            k = k.contiguous().view(k.shape[0], k.shape[1] * _num_heads, _head_dim).transpose(0, 1)
            v = v.contiguous().view(v.shape[0], v.shape[1] * _num_heads, _head_dim).transpose(0, 1)
            
            x = F.scaled_dot_product_attention(q, k, v)
            # scaled_dot_product_attention:
            # q = q / math.sqrt(_head_dim)
            # attn = torch.bmm(q, k.transpose(-2, -1))
            # attn = F.softmax(attn, dim=-1)
            # x = torch.bmm(attn, v)

            x = x.transpose(0, 1).contiguous().view(-1, _emb_dim)
            
            x = F.linear(x, _attn_out_proj_weight, _attn_out_proj_bias)
            if ssf_attn is not None:
                x = ssf_attn["attn_out"](x)

            x = x.view(_seq_len_after_vpt, _bsz, _emb_dim)

            x = x + identity

            ##########################
            ## Feed-Forward Network ##
            ##########################
            identity = x  # deep copy

            x = _ln_2(x)
            if ssf_ln is not None:
                x = ssf_ln["ln_2"](x)

            x = _mlp_in_proj(x)
            if ssf_mlp is not None:
                x = ssf_mlp["mlp_in"](x)
            
            x = _mlp_act(x)

            x = _mlp_out_proj(x)
            if ssf_mlp is not None:
                x = ssf_mlp["mlp_out"](x)
            
            if adapter is not None:
                x = x + adapter(x)
            
            if adaptformer is not None:
                x = x + adaptformer(identity)
            
            x = x + identity
            
            x = x.permute(1, 0, 2)  # LND -> NLD

        x = x[:, 0, :]
        x = self.ln_post(x)
        # x = x @ self.proj
        return x
        # if head is None:
        #     return x
        # else:
        #     return head(x)
