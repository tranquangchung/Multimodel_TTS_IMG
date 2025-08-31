# # Modified from:
# #   gpt-fast: https://github.com/pytorch-labs/gpt-fast/blob/main/generate.py
# #   DiT:      https://github.com/facebookresearch/DiT/blob/main/models.py
# import pdb
#
# import torch
# import torch.nn as nn
# from torch.nn import functional as F
# import torch._dynamo.config
# import torch._inductor.config
# import copy
# # torch._inductor.config.coordinate_descent_tuning = True
# # torch._inductor.config.triton.unique_kernel_names = True
# # torch._inductor.config.fx_graph_cache = True # Experimental feature to reduce compilation times, will be on by default in future
# from typing import Optional, Tuple, Union
# from autoregressive.models.gpt import TransformerSpeech as Transformer
# from torch.nn.attention.flex_attention import BlockMask, create_block_mask
# # create_block_mask = torch.compile(create_block_mask)
#
# default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
# def multinomial_sample_one_no_sync(probs_sort): # Does multinomial sampling without a cuda synchronization
#     q = torch.empty_like(probs_sort).exponential_(1)
#     return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)
#
# def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
#     logits = logits / max(temperature, 1e-5)
#
#     if top_k is not None:
#         v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
#         pivot = v.select(-1, -1).unsqueeze(-1)
#         logits = torch.where(logits < pivot, -float("Inf"), logits)
#     probs = torch.nn.functional.softmax(logits, dim=-1)
#     return probs
#
# def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
#     probs = logits_to_probs(logits[:, -1], temperature, top_k)
#     idx_next = multinomial_sample_one_no_sync(probs)
#     return idx_next, probs
#
# def roundup(val, multiplier):
#     return ((val - 1) // multiplier + 1) * multiplier
#
# def causal_mask(b, h, q, kv):
#     return q >= kv
#
# def prefill(model: Transformer, x: torch.Tensor, mask: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs) -> torch.Tensor:
#     # input_pos: [B, S]
#     # mask = create_block_mask(causal_mask, 1, 1, input_pos.shape[0], model.max_seq_length, device=x.device)
#     logits = model(idx=x, mask=mask, input_pos=input_pos, infer=True)
#     return sample(logits, **sampling_kwargs)[0]
#
# def decode_one_token(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, block_mask, **sampling_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
#     # input_pos: [B, 1]
#     assert input_pos.shape[-1] == 1
#     # block_index = input_pos // block_mask.BLOCK_SIZE[0]
#     # mask = block_mask[:, :, block_index]
#     # mask.mask_mod = block_mask.mask_mod
#     # mask.seq_lengths = (1, model.max_seq_length)
#     logits = model(idx=x, mask=block_mask, input_pos=input_pos, infer=True)
#     return sample(logits, **sampling_kwargs)
#
# def decode_n_tokens(model: Transformer, cur_token: torch.Tensor, mask: torch.Tensor, input_pos: torch.Tensor, num_new_tokens: int, callback=lambda _: _, **sampling_kwargs):
#     # block_mask = create_block_mask(causal_mask, 1, 1, model.max_seq_length, model.max_seq_length, device=cur_token.device)
#     new_tokens, new_probs = [], []
#     for i in range(num_new_tokens):
#         next_token, next_prob = decode_one_token(
#             model, cur_token, input_pos, mask, **sampling_kwargs
#         )
#         input_pos += 1
#         new_tokens.append(next_token.clone())
#         callback(new_tokens[-1])
#         new_probs.append(next_prob.clone())
#         cur_token = next_token.clone()
#         # print(cur_token)
#
#     return new_tokens, new_probs
#
#
# def model_forward(model, x, input_pos):
#     return model(x, input_pos)
#
# def speculative_decode(
#     model: Transformer,
#     draft_model: Transformer,
#     cur_token: torch.Tensor,
#     input_pos: int,
#     speculate_k: int,
#     **sampling_kwargs
# ) -> torch.Tensor:
#     # draft model inference sequentially
#     device = cur_token.device
#     orig_input_pos = torch.tensor([input_pos], dtype=torch.int64, device=cur_token.device)
#     draft_tokens, draft_probs = decode_n_tokens(draft_model, cur_token.view(1, -1), orig_input_pos.clone(), speculate_k, **sampling_kwargs)
#
#     draft_tokens = torch.cat(draft_tokens)
#     # parallel inference on target model using draft tokens
#     target_logits = model_forward(
#         model,
#         torch.cat([cur_token.view(1), draft_tokens]).view(1, -1),
#         torch.arange(input_pos, input_pos + speculate_k + 1, device=cur_token.device)
#     )
#     target_probs = logits_to_probs(target_logits[0], **sampling_kwargs)
#     draft_probs = torch.stack(draft_probs)
#     # q: target prob, p: draft prob
#     # q >= p: always accept draft token
#     # q < p: q/p prob to accept draft token
#     p = draft_probs[torch.arange(0, speculate_k, device=device), draft_tokens]
#     q = target_probs[torch.arange(0, speculate_k, device=device), draft_tokens]
#     accept_draft_prob = torch.minimum(torch.ones(()), q[:speculate_k]/ p)
#     rejected_locations = (torch.rand_like(accept_draft_prob) > accept_draft_prob).nonzero()
#
#     if rejected_locations.shape[0] == 0: # All draft tokens have been accepted
#         accept_length = speculate_k + 1
#         last_token = multinomial_sample_one_no_sync(target_probs[-1])
#         # fill last token into draft model
#         model_forward(
#             draft_model,
#             draft_tokens[-1].view(1, -1),
#             orig_input_pos + speculate_k,
#         )
#         return torch.cat([draft_tokens, last_token])
#     else:
#         accept_length = rejected_locations[0].item()
#         p = draft_probs[accept_length]
#         q = target_probs[accept_length]
#         new = q - p
#         new = torch.where(new > 0, new, 0.0)
#         new = new / new.sum()
#         next_token = multinomial_sample_one_no_sync(new)
#         return torch.cat([draft_tokens[:accept_length], next_token])
#
# def prepare_4d_causal_attention(
#         attention_mask,
#         sequence_length: int,
#         target_length: int,
#         batch_size: int,
#         cache_position = None,
# ):
#     device = attention_mask.device if attention_mask is not None else cache_position.device
#     dtype = attention_mask.dtype if attention_mask is not None else torch.float32
#     min_dtype = 1 #torch.finfo(dtype).min
#     causal_mask = torch.full(
#         (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
#     )
#     causal_mask = torch.triu(causal_mask, diagonal=1)
#     causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
#     invert = 1 - causal_mask
#     return invert.to(dtype=torch.bool)
#
# @torch.no_grad()
# def generate(
#     model: Transformer,
#     encoder_inputs: torch.Tensor,
#     max_new_tokens: int,
#     batch_size: int,
#     *,
#     interactive: bool,
#     draft_model: Transformer,
#     speculate_k: Optional[int] = 8,
#     callback = lambda x: x,
#     **sampling_kwargs
# ) -> torch.Tensor:
#     """
#     Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
#     """
#     is_speculative = draft_model is not None
#     # create an empty tensor of the expected final shape and fill in the current tokens
#     prompt = encoder_inputs['input_ids']
#     mask = encoder_inputs.get('attention_mask', None)
#     T = prompt.size(-1)
#     T_new = T + max_new_tokens
#     if interactive:
#         max_seq_length = 350
#     else:
#         max_seq_length = min(T_new, model.config.block_size)
#     mask = prepare_4d_causal_attention(
#         mask,
#         sequence_length=T_new, # prompt.size(-1),
#         target_length=T_new, # prompt.size(-1),
#         batch_size=batch_size,
#         cache_position=torch.arange(T_new, device=prompt.device, dtype=torch.int64),
#     )
#     device, dtype = prompt.device, prompt.dtype
#     max_seq_length = max_seq_length + speculate_k + 1 if is_speculative else max_seq_length
#     with torch.device(device):
#         model.setup_caches(max_batch_size=batch_size, max_seq_length=max_seq_length)
#         if is_speculative and draft_model is not model:
#             draft_model.setup_caches(max_batch_size=batch_size, max_seq_length=max_seq_length)
#
#     # create an empty tensor of the expected final shape and fill in the current tokens
#     empty = torch.empty(batch_size, T_new, dtype=dtype, device=device)
#     # We are just making the same prompt for every batch
#     prompt = prompt.view(1, -1).repeat(batch_size, 1)
#     empty[:, :T] = prompt
#     seq = empty
#     input_pos = torch.arange(0, T, device=device)
#
#     next_token = prefill(model, prompt.view(batch_size, -1), mask, input_pos, **sampling_kwargs).clone()
#     if is_speculative:
#         prefill(draft_model, prompt.view(batch_size, -1), input_pos, **sampling_kwargs)
#     seq[:, T] = next_token.squeeze()
#
#     input_pos = torch.tensor([T], device=device, dtype=torch.int)
#     accept_counts = [0] * (speculate_k + 1)
#
#     if is_speculative:
#         input_pos = input_pos.item()  # for speculative decoding easier to keep on host
#         while input_pos < T_new - 1:
#             cur_token = next_token.view(())
#
#             next_tokens = speculative_decode(
#                 model, draft_model, cur_token, input_pos, speculate_k, **sampling_kwargs
#             )
#
#             accept_counts[len(next_tokens) - 1] += 1
#             num_added = min(T_new - input_pos - 1, len(next_tokens))
#             seq[input_pos + 1 : input_pos + num_added + 1] = next_tokens[: num_added]
#             for i in next_tokens[: num_added,]:
#                 callback(i)
#             input_pos = input_pos + num_added
#             next_token = next_tokens[-1]
#     else:
#         generated_tokens, _ = decode_n_tokens(model, next_token.view(batch_size, -1), mask, input_pos, max_new_tokens - 1, callback=callback, **sampling_kwargs)
#         seq[:, T + 1:] = torch.cat(generated_tokens, dim=-1)
#
#     generate_stats = {
#         'accept_counts': accept_counts
#     }
#     return seq, generate_stats
#
# def encode_tokens(tokenizer, string, bos=True, device=default_device):
#     tokens = tokenizer.encode(string)
#     if bos:
#         tokens = [tokenizer.bos_id()] + tokens
#     return torch.tensor(tokens, dtype=torch.int, device=device)
#
# def _load_model(checkpoint_path, device, precision, use_tp):
#     use_cuda = 'cuda' in device
#     with torch.device('meta'):
#         model = Transformer.from_name(checkpoint_path.parent.name)
#
#     if "int8" in str(checkpoint_path):
#         print("Using int8 weight-only quantization!")
#         from quantize import WeightOnlyInt8QuantHandler
#         simple_quantizer = WeightOnlyInt8QuantHandler(model)
#         model = simple_quantizer.convert_for_runtime()
#
#     if "int4" in str(checkpoint_path):
#         print("Using int4 weight-only quantization!")
#         path_comps = checkpoint_path.name.split(".")
#         groupsize = int(path_comps[-2][1:])
#         from quantize import WeightOnlyInt4QuantHandler
#         simple_quantizer = WeightOnlyInt4QuantHandler(model, groupsize)
#         model = simple_quantizer.convert_for_runtime()
#
#     checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
#     if "model" in checkpoint and "stories" in str(checkpoint_path):
#         checkpoint = checkpoint["model"]
#     model.load_state_dict(checkpoint, assign=True)
#
#     if use_tp:
#         from tp import apply_tp
#         print("Applying tensor parallel to model ...")
#         apply_tp(model)
#
#     model = model.to(device=device, dtype=precision)
#     return model.eval()