# Modified from:
#   VQGAN:    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/transformer/mingpt.py
#   DiT:      https://github.com/facebookresearch/DiT/blob/main/models.py  
#   nanoGPT:  https://github.com/karpathy/nanoGPT/blob/master/model.py
#   llama:    https://github.com/facebookresearch/llama/blob/main/llama/model.py
#   gpt-fast: https://github.com/pytorch-labs/gpt-fast/blob/main/model.py
#   PixArt:   https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
import pdb
from dataclasses import dataclass
from typing import Optional, List
from torch import Tensor
from typing import Optional, Literal


import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.drop_path import DropPath
from typing import Callable, List, Optional, Tuple, Union
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
)
from transformers.processing_utils import Unpack
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
class KwargsForCausalLM(FlashAttentionKwargs): ...
import math
from torch.nn.attention.flex_attention import (
    _mask_mod_signature,
    BlockMask,
    flex_attention,
)
from autoregressive.models.generate import generate as generate_img_fn

def find_multiple(n: int, k: int):
    if n % k == 0:
        return n
    return n + k - (n % k)

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layer: int = 32
    n_head: int = 32
    n_kv_head: Optional[int] = None
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    rope_base: float = 10000
    norm_eps: float = 1e-5
    initializer_range: float = 0.02
    
    token_dropout_p: float = 0.1
    attn_dropout_p: float = 0.0
    resid_dropout_p: float = 0.1
    ffn_dropout_p: float = 0.1
    drop_path_rate: float = 0.0

    num_classes: int = 1000
    caption_dim: int = 2048
    class_dropout_prob: float = 0.1
    model_type: str = 'c2i'

    vocab_size: int = 16384
    cls_token_num: int = 1
    block_size: int = 256
    max_batch_size: int = 32
    max_seq_len: int = 2048


#################################################################################
#                      Embedding Layers for Class Labels                        #
#################################################################################
class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels).unsqueeze(1)
        return embeddings


#################################################################################
#                      Embedding Layers for Text Feature                        #
#################################################################################
class CaptionEmbedder(nn.Module):
    """
    Embeds text caption into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, in_channels, hidden_size, uncond_prob, token_num=120):
        super().__init__()
        self.cap_proj = MLP(in_features=in_channels, hidden_features=hidden_size, out_features=hidden_size)
        self.register_buffer("uncond_embedding", nn.Parameter(torch.randn(token_num, in_channels) / in_channels ** 0.5))
        self.uncond_prob = uncond_prob

    def token_drop(self, caption, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(caption.shape[0], device=caption.device) < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        caption = torch.where(drop_ids[:, None, None], self.uncond_embedding, caption)
        return caption

    def forward(self, caption, train, force_drop_ids=None):
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            caption = self.token_drop(caption, force_drop_ids)
        embeddings = self.cap_proj(caption)
        return embeddings


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


#################################################################################
#                                  GPT Model                                    #
#################################################################################
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        hidden_dim = 4 * config.dim
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if config.ffn_dim_multiplier is not None:
            hidden_dim = int(config.ffn_dim_multiplier * hidden_dim)
        hidden_dim = find_multiple(hidden_dim, config.multiple_of)

        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)
        self.ffn_dropout = nn.Dropout(config.ffn_dropout_p)

    def forward(self, x):
        return self.ffn_dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_head, head_dim, dtype):
        super().__init__()
        cache_shape = (max_batch_size, n_head, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0
        self.dim = config.dim
        self.head_dim = config.dim // config.n_head
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head if config.n_kv_head is not None else config.n_head
        total_kv_dim = (self.n_head + 2 * self.n_kv_head) * self.head_dim

        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_kv_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        # regularization
        self.attn_dropout_p = config.attn_dropout_p
        self.resid_dropout = nn.Dropout(config.resid_dropout_p)

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor = None, 
        input_pos: Optional[torch.Tensor] = None, 
        mask: Optional[torch.Tensor] = None
    ):
        bsz, seqlen, _ = x.shape
        kv_size = self.n_kv_head * self.head_dim
        xq, xk, xv = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        
        xq = apply_rotary_emb(xq, freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis)

        xq, xk, xv = map(lambda x: x.transpose(1, 2), (xq, xk, xv))

        if self.kv_cache is not None:
            keys, values = self.kv_cache.update(input_pos, xk, xv)
        else:
            keys, values = xk, xv
        keys = keys.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        values = values.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        output = F.scaled_dot_product_attention(
            xq, keys, values, 
            attn_mask=mask, 
            is_causal=True if mask is None else False, # is_causal=False is for KV cache
            dropout_p=self.attn_dropout_p if self.training else 0)            
        
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        output = self.resid_dropout(self.wo(output))
        return output


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs, drop_path: float):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, start_pos: int, mask: Optional[torch.Tensor] = None):
        h = x + self.drop_path(self.attention(self.attention_norm(x), freqs_cis, start_pos, mask))
        out = h + self.drop_path(self.feed_forward(self.ffn_norm(h)))
        return out


class Transformer(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layer = config.n_layer
        self.block_size = config.block_size
        self.num_classes = config.num_classes
        self.model_type = config.model_type
        self.cls_token_num = config.cls_token_num
        if self.model_type == 'c2i':
            self.cls_embedding = LabelEmbedder(config.num_classes, config.dim, config.class_dropout_prob)
        elif self.model_type == 't2i':
            self.cls_embedding = CaptionEmbedder(config.caption_dim, config.dim, config.class_dropout_prob)
        else:
            raise Exception("please check model type")
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.tok_dropout = nn.Dropout(config.token_dropout_p)

        # transformer blocks
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.n_layer)]
        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layer):
            self.layers.append(TransformerBlock(config, dpr[layer_id]))

        # output layer
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # 2d rotary pos embedding
        grid_size = int(self.block_size ** 0.5)
        assert grid_size * grid_size == self.block_size
        self.freqs_cis = precompute_freqs_cis_2d(grid_size, self.config.dim // self.config.n_head, self.config.rope_base, self.cls_token_num)
        
        # KVCache
        self.max_batch_size = -1
        self.max_seq_length = -1

        self.initialize_weights()

    def initialize_weights(self):        
        # Initialize nn.Linear and nn.Embedding
        self.apply(self._init_weights)

        # Zero-out output layers:
        nn.init.constant_(self.output.weight, 0)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def setup_caches(self, max_batch_size, max_seq_length, dtype):
        # if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
        #     return
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        for b in self.layers:
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_head, head_dim, dtype)

        causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool))
        self.causal_mask = causal_mask.unsqueeze(0).repeat(self.max_batch_size, 1, 1)
        grid_size = int(self.config.block_size ** 0.5)
        assert grid_size * grid_size == self.block_size
        self.freqs_cis = precompute_freqs_cis_2d(grid_size, self.config.dim // self.config.n_head, self.config.rope_base, self.cls_token_num)

    def forward(
        self, 
        idx: torch.Tensor, 
        cond_idx: torch.Tensor,  # cond_idx_or_embed
        input_pos:  Optional[torch.Tensor] = None, 
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
    ):
        if idx is not None and cond_idx is not None: # training or naive inference
            cond_embeddings = self.cls_embedding(cond_idx, train=self.training)[:,:self.cls_token_num]
            token_embeddings = self.tok_embeddings(idx)
            token_embeddings = torch.cat((cond_embeddings, token_embeddings), dim=1)
            h = self.tok_dropout(token_embeddings)
            self.freqs_cis = self.freqs_cis.to(h.device)
        else: # inference
            if cond_idx is not None: # prefill in inference
                token_embeddings = self.cls_embedding(cond_idx, train=self.training)[:,:self.cls_token_num]
            else: # decode_n_tokens(kv cache) in inference
                token_embeddings = self.tok_embeddings(idx)
            
            bs = token_embeddings.shape[0]
            mask = self.causal_mask[:bs, None, input_pos]
            h = self.tok_dropout(token_embeddings)
            self.freqs_cis = self.freqs_cis
        if self.training:
            freqs_cis = self.freqs_cis[:token_embeddings.shape[1]]
        else:
            freqs_cis = self.freqs_cis[input_pos]
        # transformer blocks
        #print(f"h: {h.shape}, freqs_cis: {freqs_cis.shape}, input_pos: {input_pos}, mask: {mask.shape}")
        for layer in self.layers:
            h = layer(h, freqs_cis, input_pos, mask)
        
        # output layers
        h = self.norm(h)
        logits = self.output(h).float()
        
        if self.training:
            logits = logits[:, self.cls_token_num - 1:].contiguous()

        # if we are given some desired targets also calculate the loss
        loss = None
        if valid is not None:
            loss_all = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')
            valid_all = valid[:,None].repeat(1, targets.shape[1]).view(-1)
            loss = (loss_all * valid_all).sum() / max(valid_all.sum(), 1)
        elif targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss


    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        return list(self.layers)

class TransformerSpeech(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layer = config.n_layer
        self.block_size = config.block_size
        self.num_classes = config.num_classes
        self.model_type = config.model_type
        self.cls_token_num = config.cls_token_num

        self.text_embeddings = nn.Embedding(self.vocab_size, config.dim)

        # transformer blocks
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.n_layer)]
        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layer):
            self.layers.append(TransformerBlock(config, dpr[layer_id]))

        # output layer
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.config.vocab_speech_size = 1002
        self.speech_output = nn.Linear(config.dim, self.config.vocab_speech_size, bias=False)

        max_seq_len = 2048
        self.freqs_cis = precompute_freqs_cis(
            seq_len=max_seq_len,
            n_elem=config.dim // config.n_head,
            base=10000,
            dtype=torch.float32,
            rope_scaling=None,  # No scaling for now
        )
        print("Shape of freqs_cis:", self.freqs_cis.shape)

        # KVCache
        self.max_batch_size = -1
        self.max_seq_length = -1

    def forward(
            self,
            idx: torch.Tensor,
            targets: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None,
            input_pos: Optional[Tensor] = None,
            **kwargs,
    ):
        assert self.freqs_cis is not None, "Caches must be initialized first"
        h = self.text_embeddings(idx)
        # Use 1D RoPE
        seq_len = h.shape[1]
        freqs_cis = self.freqs_cis[:seq_len].to(h.device)
        # Forward through layers
        for layer in self.layers:
            if targets is not None:
                h = layer(h, freqs_cis, input_pos, None)
            else:
                h = layer(h, freqs_cis, input_pos, mask)
        h = self.norm(h)
        logits = self.speech_output(h).float()

        loss = None
        if targets is not None:
            # loss = ForCausalLMLoss(logits=logits, labels=targets, vocab_size=self.config.vocab_speech_size, **kwargs)
            logits = logits[:, :-1, :].contiguous()
            targets = targets[:, 1:].contiguous()
            loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]), targets.view(-1), ignore_index=-100
            )
        else:
            return logits

        #######################################
        return_dict = {
            "loss": loss,
            "logits": logits,
        }
        return return_dict

    @torch.no_grad()
    def generate(
        self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, min_p=None
    ):
        for _ in range(max_new_tokens):
            context = (
                idx
                if idx.size(1) < self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            logits = self(context)

            logits = logits[:, -1, :] / temperature

            if top_p is not None and top_p > 0.0:
                probs = torch.softmax(logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(
                    probs, descending=True, dim=-1
                )
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                mask = cumulative_probs >= top_p
                mask[..., 0] = True

                cutoff_indices = mask.int().argmax(dim=-1, keepdim=True)

                top_p_mask = torch.zeros_like(logits, dtype=torch.bool)
                for b in range(logits.size(0)):
                    cut = cutoff_indices[b].item()
                    kept_indices = sorted_indices[b, : cut + 1]
                    top_p_mask[b, kept_indices] = True
                logits[~top_p_mask] = float("-inf")

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if min_p is not None and min_p > 0.0:
                logit_max = logits.max(dim=-1, keepdim=True).values
                threshold = logit_max + torch.log(
                    torch.tensor(min_p, device=logits.device, dtype=logits.dtype)
                )
                logits[logits < threshold] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            if idx_next == 1001:
                break
            idx = torch.cat([idx, idx_next], dim=-1)

        return idx

    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        return list(self.layers)

class LoRALinear(nn.Module):
    """
    LoRA wrapper for Linear layer.
    """
    def __init__(self, linear_layer: nn.Linear, rank: int = 4, alpha: float = 8.0, dropout: float = 0.0):
        super().__init__()
        self.linear = linear_layer
        self.rank = rank
        self.alpha = alpha
        self.enable_lora = True
        if rank > 0:
            self.lora_A = nn.Linear(linear_layer.in_features, rank, bias=False)
            self.lora_B = nn.Linear(rank, linear_layer.out_features, bias=False)
            self.scaling = alpha / rank
            nn.init.normal_(self.lora_A.weight, std=0.02)
            nn.init.zeros_(self.lora_B.weight)
            self.dropout = nn.Dropout(dropout)
        else:
            self.lora_A = self.lora_B = None

    def forward(self, x):
        if x.dtype != self.linear.weight.dtype:
            x = x.to(self.linear.weight.dtype)
        out = self.linear(x)
        if self.enable_lora and self.lora_A is not None and self.lora_B is not None:
            out = out + self.lora_B(self.dropout(self.lora_A(x))) * self.scaling
        return out


class MultiTaskImageSpeech(nn.Module):
    def __init__(
        self,
        pretrained_image_model: nn.Module,
        text_vocab_size: int = 16384,
        speech_vocab_size: int = 1002,
        n_speech_extra_layers: int = 4,
        share_kv_cache: bool = False,
        image_backbone_tuning_mode: Literal["frozen", "finetune", "lora"] = "frozen",
        lora_rank: int = 4,
        lora_alpha: float = 8.0,
        lora_dropout: float = 0.0,
        lora_targets: list = None,
        residual_connection: bool = False,
    ):
        super().__init__()
        self.img = pretrained_image_model
        self.config = self.img.config
        self.vocab_img_size = self.img.vocab_size
        self.vocab_text_size = text_vocab_size
        self.vocab_speech_size = speech_vocab_size
        self.n_head = self.config.n_head
        self.head_dim = self.config.dim // self.n_head
        self.image_backbone_tuning_mode = image_backbone_tuning_mode
        self.residual_connection = True
        print("Use residual connection:", self.residual_connection)

        # Default LoRA target modules
        if lora_targets is None:
            lora_targets = [
                "attention.wqkv", "attention.wo",
                "feed_forward.w1", "feed_forward.w2", "feed_forward.w3",
            ]
        # ------------- Image Backbone Handling -------------
        if image_backbone_tuning_mode == "frozen":
            for p in self.img.parameters():
                p.requires_grad = False
            self.img.eval()

        elif image_backbone_tuning_mode == "finetune":
            # Unfreeze all
            for p in self.img.parameters():
                p.requires_grad = True
            self.img.train()  # or keep eval if desired

        elif image_backbone_tuning_mode == "lora":
            for p in self.img.parameters():
                p.requires_grad = False  # freeze original

            # Apply LoRA to specified layers
            for name, module in self.img.named_modules():
                if any(tgt in name for tgt in lora_targets):
                    if isinstance(module, nn.Linear):
                        print(f"Applying LoRA to {name}")
                        parent_name = ".".join(name.split(".")[:-1])
                        leaf_name = name.split(".")[-1]
                        lora_linear = LoRALinear(
                            module,
                            rank=lora_rank,
                            alpha=lora_alpha,
                            dropout=lora_dropout
                        )
                        # Replace in parent
                        parent = self.img
                        for part in parent_name.split("."):
                            if hasattr(parent, part):
                                parent = getattr(parent, part)
                        setattr(parent, leaf_name, lora_linear)
            self.img.train()
            # Ensure LoRA layers are trainable
            for n, p in self.img.named_parameters():
                if "lora" in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False

        else:
            raise ValueError(f"Unsupported tuning mode: {image_backbone_tuning_mode}")

        # ------------- Speech path -------------
        self.text_embeddings = nn.Embedding(self.vocab_text_size, self.config.dim)

        dpr = [x.item() for x in torch.linspace(0, self.config.drop_path_rate, n_speech_extra_layers)]
        self.speech_layers = nn.ModuleList(
            [TransformerBlock(self.config, dpr[i]) for i in range(n_speech_extra_layers)]
        )
        self.speech_norm = RMSNorm(self.config.dim, eps=self.config.norm_eps)
        self.speech_head = nn.Linear(self.config.dim, self.vocab_speech_size, bias=False)

        max_seq_len = max(self.config.max_seq_len, 2048)
        self.speech_freqs_cis = precompute_freqs_cis(
            seq_len=max_seq_len,
            n_elem=self.config.dim // self.config.n_head,
            base=10000,
            dtype=torch.float32,
            rope_scaling=None,
        )
        print("Shape of freqs_cis:", self.speech_freqs_cis.shape)

    def _set_lora(self, enabled: bool):
        for m in self.img.modules():
            if isinstance(m, LoRALinear):
                m.enable_lora = enabled

    def _clear_kv_cache(self):
        for bl in self.img.layers:
            bl.attention.kv_cache = None

    # -------------------- Speech API --------------------
    def speech_forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        p = next(self.img.parameters())
        h = self.text_embeddings(idx).to(device=p.device, dtype=p.dtype)
        seq_len = h.shape[1]
        freqs_cis = self.speech_freqs_cis[:seq_len].to(device=h.device, dtype=h.dtype)

        # Forward through image backbone (with grad depending on mode)
        if self.image_backbone_tuning_mode != "frozen":
            # Allow gradients if finetune or LoRA
            for layer in self.img.layers:
                h = layer(h, freqs_cis, input_pos, mask)
        else:
            # Frozen: no grad
            with torch.no_grad():
                for layer in self.img.layers:
                    h = layer(h, freqs_cis, input_pos, mask)

        # Extra speech-only layers
        for layer in self.speech_layers:
            h = layer(h, freqs_cis, input_pos, mask)

        h = self.speech_norm(h)
        logits = self.speech_head(h).float()

        loss = None
        if targets is not None:
            logits_shift = logits[:, :-1, :].contiguous()
            targets_shift = targets[:, 1:].contiguous()
            loss = F.cross_entropy(
                logits_shift.view(-1, logits_shift.size(-1)),
                targets_shift.view(-1),
                ignore_index=-100
            )

        return {"loss": loss, "logits": logits}

    @torch.no_grad()
    def speech_generate(
        self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, min_p=None
    ):
        """
        Minimal sampler that uses speech_forward() with no targets.
        """
        self._clear_kv_cache()
        self._set_lora(True)
        for _ in range(max_new_tokens):
            context = idx if idx.size(1) < self.config.block_size else idx[:, -self.config.block_size:]
            out = self.speech_forward(context)
            logits = out["logits"][:, -1, :] / max(temperature, 1e-6)

            if top_p is not None and top_p > 0.0:
                probs = torch.softmax(logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                mask = cumulative_probs >= top_p
                mask[..., 0] = True
                cutoff = mask.int().argmax(dim=-1, keepdim=True)
                top_p_mask = torch.zeros_like(logits, dtype=torch.bool)
                for b in range(logits.size(0)):
                    k = cutoff[b].item()
                    kept = sorted_indices[b, : k + 1]
                    top_p_mask[b, kept] = True
                logits[~top_p_mask] = float("-inf")

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if min_p is not None and min_p > 0.0:
                m = logits.max(dim=-1, keepdim=True).values
                thr = m + torch.log(torch.tensor(min_p, device=logits.device, dtype=logits.dtype))
                logits[logits < thr] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Stop if EOS token is generated
            if idx_next.item() == self.vocab_speech_size - 1:
                break

            idx = torch.cat([idx, idx_next], dim=-1)

        return idx

    # -------------------- Image API --------------------
    @torch.no_grad()
    def image_forward(
        self,
        idx: torch.Tensor,
        cond_idx: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
    ):
        self.img.eval()  # ensure eval mode for consistency
        return self.img(
            idx=idx, cond_idx=cond_idx, input_pos=input_pos,
            targets=targets, mask=mask, valid=valid
        )


    def image_generate(
        self,
        cond: torch.Tensor,
        max_new_tokens: int,
        emb_masks: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
        cfg_interval: int = -1,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        sample_logits: bool = True,
    ) -> torch.Tensor:
        """
        """
        self._clear_kv_cache()
        self._set_lora(False)
        device = next(self.img.parameters()).device
        cond = cond.to(device=device)
        if emb_masks is not None:
            emb_masks = emb_masks.to(device)

        self.img.eval()

        out = generate_img_fn(
            model=self.img,
            cond=cond,
            max_new_tokens=max_new_tokens,
            emb_masks=emb_masks,
            cfg_scale=cfg_scale,
            cfg_interval=cfg_interval,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            sample_logits=sample_logits,
        )
        return out

def fixed_cross_entropy(source, target, num_items_in_batch: int = None, ignore_index: int = -100, reduction = "mean", **kwargs):
    # reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction, label_smoothing=0.1)
    if reduction == "sum":
        loss = loss / num_items_in_batch
    return loss


def ForCausalLMLoss(
    logits, labels, vocab_size: int, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs
):
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()
    labels = labels.to(logits.device)
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    batch_size, seq_len = shift_labels.shape

    # Flatten the tokens
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss_1 = fixed_cross_entropy(shift_logits, shift_labels, num_items_in_batch, ignore_index, "mean", **kwargs)
    return loss_1


#################################################################################
#                      Rotary Positional Embedding Functions                    #
#################################################################################
# https://github.com/pytorch-labs/gpt-fast/blob/main/model.py 
def precompute_freqs_cis(
    seq_len: int, n_elem: int, base: int = 10000,
    dtype: torch.dtype = torch.bfloat16,
    rope_scaling: Optional[dict] = None,
) -> Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    if rope_scaling is not None:
        freqs = apply_rope_scaling(freqs, rope_scaling)
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=dtype)

def precompute_freqs_cis_2d(grid_size: int, n_elem: int, base: int = 10000, cls_token_num=120):
    # split the dimension into half, one for x and one for y
    half_dim = n_elem // 2
    freqs = 1.0 / (base ** (torch.arange(0, half_dim, 2)[: (half_dim // 2)].float() / half_dim))
    t = torch.arange(grid_size, device=freqs.device)
    freqs = torch.outer(t, freqs) # (grid_size, head_dim // 2)
    freqs_grid = torch.concat([
        freqs[:, None, :].expand(-1, grid_size, -1),
        freqs[None, :, :].expand(grid_size, -1, -1),
    ], dim=-1)  # (grid_size, grid_size, head_dim // 2)
    cache_grid = torch.stack([torch.cos(freqs_grid), torch.sin(freqs_grid)], dim=-1) # (grid_size, grid_size, head_dim // 2, 2)
    cache = cache_grid.flatten(0, 1)
    cond_cache = torch.cat([torch.zeros(cls_token_num, n_elem // 2, 2), cache]) # (cls_token_num+grid_size**2, head_dim // 2, 2)
    return cond_cache 


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    # x: (bs, seq_len, n_head, head_dim)
    # freqs_cis (seq_len, head_dim // 2, 2)
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2) # (bs, seq_len, n_head, head_dim//2, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2) # (1, seq_len, 1, head_dim//2, 2)
    x_out2 = torch.stack([
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
    ], dim=-1)
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)

def apply_rope_scaling(freqs: torch.Tensor, rope_scaling: Optional[dict] = None):
    factor = rope_scaling["factor"]
    low_freq_factor = rope_scaling["low_freq_factor"]
    high_freq_factor = rope_scaling["high_freq_factor"]
    old_context_len = rope_scaling["original_max_position_embeddings"]

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            new_freqs.append((1 - smooth) * freq / factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)



#################################################################################
#                                GPT Configs                                    #
#################################################################################
### text-conditional
def GPT_7B(**kwargs):
    return Transformer(ModelArgs(n_layer=32, n_head=32, dim=4096, **kwargs)) # 6.6B

def GPT_3B(**kwargs):
    return Transformer(ModelArgs(n_layer=24, n_head=32, dim=3200, **kwargs)) # 3.1B

def GPT_1B(**kwargs):
    return Transformer(ModelArgs(n_layer=22, n_head=32, dim=2048, **kwargs)) # 1.2B

### class-conditional
def GPT_XXXL(**kwargs):
    return Transformer(ModelArgs(n_layer=48, n_head=40, dim=2560, **kwargs)) # 3.9B

def GPT_XXL(**kwargs):
    return Transformer(ModelArgs(n_layer=48, n_head=24, dim=1536, **kwargs)) # 1.4B

def GPT_XL(**kwargs):
    return Transformer(ModelArgs(n_layer=36, n_head=20, dim=1280, **kwargs)) # 775M

def GPT_L(**kwargs):
    return Transformer(ModelArgs(n_layer=24, n_head=16, dim=1024, **kwargs)) # 343M

def GPT_B(**kwargs):
    return Transformer(ModelArgs(n_layer=12, n_head=12, dim=768, **kwargs)) # 111M

def GPT_XXL_speech(**kwargs):
    # return TransformerSpeech(ModelArgs(n_layer=36, n_head=20, dim=1280, **kwargs)) # 1.4B
    return TransformerSpeech(ModelArgs(n_layer=8, n_head=20, dim=1280, **kwargs)) # 1.4B

def GPT_Small_speech(**kwargs):
    return TransformerSpeech(ModelArgs(n_layer=2, n_head=20, dim=1280, **kwargs)) # 1.4B

GPT_models = {
    'GPT-B': GPT_B, 'GPT-L': GPT_L, 'GPT-XL': GPT_XL, 'GPT-XXL': GPT_XXL, 'GPT-XXXL': GPT_XXXL,
    'GPT-1B': GPT_1B, 'GPT-3B': GPT_3B, 'GPT-7B': GPT_7B,
    'GPT-XXL_speech': GPT_XXL_speech,
    'GPT-Small_speech': GPT_Small_speech,
}