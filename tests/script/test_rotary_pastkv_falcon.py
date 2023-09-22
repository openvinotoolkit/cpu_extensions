# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import math
import sys
import torch
import numpy as np
import llmdnn as ld
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import skip_init
from typing import Optional, Tuple, Union, List, Callable, Dict, Any

# copy from transformers/models/falcon/modeling_falcon.py
# rotary pos emb helpers (torch.jit.script does not seem to support staticmethod...)
def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class FalconRotaryEmbedding(nn.Module):
    """Implementation of RotaryEmbedding from GPT-NeoX.
    This implementation is designed to operate on queries and keys that are compatible with `[batch_size,
    n_heads_per_partition, seq_len, head_dim]` (e.g. MinGPTAttention format).
    """

    def __init__(self, head_dim: int, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.head_dim = head_dim
        self.seq_len_cached = -1
        self.cos_cached: torch.Tensor | None = None
        self.sin_cached: torch.Tensor | None = None

    def cos_sin(self, seq_len: int, past_key_values_length: int, device="cpu", dtype=torch.bfloat16) -> torch.Tensor:
        total_length = seq_len + past_key_values_length
        if total_length > self.seq_len_cached:
            self.seq_len_cached = total_length
            t = torch.arange(total_length, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(device)

            if dtype in [torch.float16, torch.bfloat16]:
                emb = emb.float()

            self.cos_cached = emb.cos()[None, :, :]
            self.sin_cached = emb.sin()[None, :, :]

            # self.cos_cached = self.cos_cached.type(dtype)
            # self.sin_cached = self.sin_cached.type(dtype)

        return (
            self.cos_cached[:, past_key_values_length : seq_len + past_key_values_length],
            self.sin_cached[:, past_key_values_length : seq_len + past_key_values_length],
        )

    def forward(self, query, key, past_key_values_length=0):
        batch, seq_len, head_dim = query.shape
        cos, sin = self.cos_sin(seq_len, past_key_values_length, query.device, query.dtype)
        return (query * cos) + (rotate_half(query) * sin), (key * cos) + (rotate_half(key) * sin)

class FalconAttention(torch.nn.Module):
    def __init__(self, hidden_size, num_attention_heads, num_kv_heads, new_decoder_architecture=True):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.split_size = self.hidden_size

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"`hidden_size` must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.maybe_rotary = FalconRotaryEmbedding(self.head_dim) #if config.rotary else lambda q, k, t: (q, k)

        # Layer-wise attention scaling
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.beta = self.inv_norm_factor
        if new_decoder_architecture:
            qkv_out_dim = (num_kv_heads * 2 + num_attention_heads) * self.head_dim
        elif config.multi_query:
            qkv_out_dim = self.hidden_size + 2 * self.head_dim
        else:
            qkv_out_dim = 3 * self.hidden_size
        self.new_decoder_architecture = new_decoder_architecture
        self.num_kv_heads = num_kv_heads #if (self.new_decoder_architecture or not self.multi_query) else 1

    def _split_heads(self, fused_qkv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split the last dimension into (num_heads, head_dim), results share same memory storage as `fused_qkv`

        Args:
            fused_qkv (`torch.tensor`, *required*): [batch_size, seq_length, num_heads * 3 * head_dim]

        Returns:
            query: [batch_size, seq_length, num_heads, head_dim] key: [batch_size, seq_length, num_heads, head_dim]
            value: [batch_size, seq_length, num_heads, head_dim]
        """
        if self.new_decoder_architecture:
            batch, seq_len, _ = fused_qkv.shape
            qkv = fused_qkv.view(batch, seq_len, -1, self.num_heads // self.num_kv_heads + 2, self.head_dim)
            query = qkv[:, :, :, :-2]
            key = qkv[:, :, :, [-2]]
            value = qkv[:, :, :, [-1]]
            key = torch.broadcast_to(key, query.shape)
            value = torch.broadcast_to(value, query.shape)

            query, key, value = [x.flatten(2, 3) for x in (query, key, value)]
            return query, key, value
        elif not self.multi_query:
            batch_size, seq_length, three_times_hidden_size = fused_qkv.shape
            fused_qkv = fused_qkv.view(batch_size, seq_length, self.num_heads, 3, self.head_dim)
            return fused_qkv[..., 0, :], fused_qkv[..., 1, :], fused_qkv[..., 2, :]
        else:
            batch_size, seq_length, three_times_hidden_size = fused_qkv.shape
            fused_qkv = fused_qkv.view(batch_size, seq_length, self.num_heads + 2, self.head_dim)
            return fused_qkv[..., :-2, :], fused_qkv[..., [-2], :], fused_qkv[..., [-1], :]

    def forward(
        self,
        fused_qkv: torch.Tensor,        # [batch_size, seq_length, 9216]
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        num_kv_heads = self.num_heads if self.new_decoder_architecture else self.num_kv_heads
        # 3 x [batch_size, seq_length, num_heads, head_dim]
        (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)

        batch_size, query_length, _, _ = query_layer.shape

        query_layer = query_layer.transpose(1, 2).reshape(batch_size * self.num_heads, query_length, self.head_dim)
        key_layer = key_layer.transpose(1, 2).reshape(
            batch_size * num_kv_heads,
            query_length,
            self.head_dim,
        )
        value_layer = value_layer.transpose(1, 2).reshape(batch_size * num_kv_heads, query_length, self.head_dim)

        layer_past = [item.view(item.size(0) * item.size(1), item.size(2), item.size(3)) for item in layer_past]
        past_kv_length = 0 if layer_past is None else layer_past[0].shape[1]
        query_layer, key_layer = self.maybe_rotary(query_layer, key_layer, past_kv_length)

        if layer_past is not None:
            past_key, past_value = layer_past
            # concatenate along seq_length dimension:
            #  - key: [batch_size * self.num_heads, kv_length, head_dim]
            #  - value: [batch_size * self.num_heads, kv_length, head_dim]
            key_layer = torch.cat((past_key, key_layer), dim=1)
            value_layer = torch.cat((past_value, value_layer), dim=1)

        _, kv_length, _ = key_layer.shape
        if use_cache:
            present = (key_layer, value_layer)
        else:
            present = None

        query_layer_ = query_layer.reshape(batch_size, self.num_heads, -1, self.head_dim)
        key_layer_ = key_layer.reshape(batch_size, num_kv_heads, -1, self.head_dim)
        value_layer_ = value_layer.reshape(batch_size, num_kv_heads, -1, self.head_dim)

        return query_layer_, key_layer_, value_layer_


class FalconAttentionExt:
    def __init__(self, num_attention_heads, hidden_size, max_position_embeddings, rotary_ndims, rotary_emb_base=10000):
        num_heads = num_attention_heads
        head_size = hidden_size // num_attention_heads
        max_seq_len = max_position_embeddings

        inv_freq = 1. / (rotary_emb_base ** (torch.arange(0, rotary_ndims, 2).float() / rotary_ndims))
        #inv_freq = inv_freq.half()
        self.max_seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None
        # use f32 to pass accuracy test
        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=inv_freq.device, dtype=inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()[None, None, :, :]
        self.sin_cached = emb.sin()[None, None, :, :]

    # qkv: [batch, seq_len, ((num_heads + num_kv_heads * 2) * head_size)]
    # layer_past_padded: [batch, num_attention_heads, MAX_SEQ_LEN, head_size_aligned]
    # past_seq_len: past_seq_len==layer_past.shape[-2]
    # return:
    #       0: (k, v): ([batch, num_attention_heads, MAX_SEQ_LEN, head_size_aligned], [batch, num_attention_heads, MAX_SEQ_LEN, head_size_aligned])
    #       1: query: [batch, num_attention_heads, seq_len, head_size_aligned]
    #       2: k: [batch, num_attention_heads, MAX_SEQ_LEN, head_size_aligned]
    #       3: v: [batch, num_attention_heads, MAX_SEQ_LEN, head_size_aligned]
    def forward(self, qkv, num_kv_heads, k_past, v_past):
        return ld.emb_gpt(qkv, num_kv_heads, k_past, v_past, self.cos_cached, self.sin_cached)


HEAD_NUM = 128
SIZE_PER_HEAD = 64
SIZE_PER_HEAD_ALIGN = 64
NUM_KV_HEADS = 8
HIDDEN_SIZE = HEAD_NUM * SIZE_PER_HEAD
MAX_POSITION_EMBEDDINGS = 1024 #2048
ROTARY_EMB_BASE = 10000
ROTARY_PCT = 0.5
MAX_SEQ_LEN = 1024
def get_ref_model():
    ref_net = FalconAttention(hidden_size=HIDDEN_SIZE, num_attention_heads=HEAD_NUM, num_kv_heads=NUM_KV_HEADS, new_decoder_architecture=True)
    ref_net.maybe_rotary.cos_sin(0, MAX_SEQ_LEN)
    ref_net = ref_net.to(dtype=torch.bfloat16)
    return ref_net

def test_falcon():
    inputs = [
        # qkv: [batch, seq_len, (num_heads + 2 * num_kv_heads) * head_size)]
        # layer_past: [batch, num_attention_heads, past_seq_len, head_size]
        (np.random.random(size=[2, 200, (HEAD_NUM + 2 * NUM_KV_HEADS) * SIZE_PER_HEAD]).astype(np.float32),
         np.random.random(size=[2, HEAD_NUM, 0, SIZE_PER_HEAD]).astype(np.float32),
         np.random.random(size=[2, HEAD_NUM, 0, SIZE_PER_HEAD]).astype(np.float32)),
        (np.random.random(size=[2, 1, (HEAD_NUM + 2 * NUM_KV_HEADS) * SIZE_PER_HEAD]).astype(np.float32),
         np.random.random(size=[2, HEAD_NUM, 200, SIZE_PER_HEAD]).astype(np.float32),
         np.random.random(size=[2, HEAD_NUM, 200, SIZE_PER_HEAD]).astype(np.float32)),
    ]
    ref_net = get_ref_model()
    net_seq = FalconAttentionExt(HEAD_NUM, HIDDEN_SIZE, MAX_POSITION_EMBEDDINGS, SIZE_PER_HEAD, ROTARY_EMB_BASE)
    with torch.cpu.amp.autocast():
        for (i, input) in enumerate(inputs):
            qkv, layer_past_key, layer_past_value = input
            qkv = torch.from_numpy(qkv).to(torch.bfloat16)
            layer_past_key = torch.from_numpy(layer_past_key).to(torch.bfloat16)
            layer_past_value = torch.from_numpy(layer_past_value).to(torch.bfloat16)

            query_ref, key_ref, value_ref = ref_net.forward(qkv, (layer_past_key, layer_past_value))
            query_ref = query_ref.to(dtype=torch.bfloat16)
            key_ref = key_ref.to(dtype=torch.bfloat16)
            
            # no prealloc past kv
            query, key, value = net_seq.forward(qkv, NUM_KV_HEADS, layer_past_key, layer_past_value)
            # check query
            if not torch.allclose(query_ref, query, rtol=0.001, atol=0.01):
                print(f"error at sequence query index {i} ref:\n{query_ref} \ncur:\n {query} ")
                assert(False)
            # check key
            if not torch.allclose(key_ref, key, rtol=0.001, atol=0.01):
                print(f"error at sequence key index {i} ref:\n{key_ref} \ncur:\n {key} ")
                assert(False)
            # check value
            if not torch.allclose(value_ref, value, rtol=0.001, atol=0.01):
                print(f"error at sequence value index {i} ref:\n{value_ref} \ncur:\n {value} ")
                assert(False)

    print('done.')
    return

if __name__ == "__main__":
    test_falcon()