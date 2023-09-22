# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import math
import sys
import torch
import numpy as np
import llmdnn as ld
from torch import nn

# copy from transformers/models/gpt_neox/modeling_gpt_neox.py
class GPTNeoXAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_attention_heads
        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
        )
        self.norm_factor = torch.sqrt(torch.tensor(self.head_size, dtype=torch.float32)).to(torch.get_default_dtype())

    def forward(self, query, key, value, attention_mask, q_quant=None, k_quant=None, qk_quant=None, v_quant=None, requant=None):
        if q_quant:
            # quant
            query = query.to(torch.float32)
            key = key.to(torch.float32)
            value = value.to(torch.float32)

        # Compute attention
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, q_quant, k_quant, qk_quant, v_quant)
        if q_quant:
            attn_output = (attn_output * requant).round().clamp(-128, 127).to(torch.int8)

        # Reshape outputs
        attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_size)

        return attn_output

    @classmethod
    def _merge_heads(cls, tensor, num_attention_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden dim
        """
        # tensor [bs, num_attention_heads, seq_len, attn_head_size]
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        # -> [bs, seq_len, num_attention_heads, attn_head_size]
        tensor = tensor.view(tensor.size(0), tensor.size(1), num_attention_heads * attn_head_size)
        # -> [bs, seq_len, hidden_size]
        return tensor

    def _attn(self, query, key, value, attention_mask, q_quant, k_quant, qk_quant, v_quant):
        # q, k, v: [bs, num_attention_heads, seq_len, attn_head_size]
        # compute causal mask from causal mask buffer
        batch_size, num_attention_heads, query_length, attn_head_size = query.size()
        key_length = key.size(-2)

        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]

        query = query.view(batch_size * num_attention_heads, query_length, attn_head_size)
        key = key.view(batch_size * num_attention_heads, key_length, attn_head_size)
        attn_scores = torch.zeros(
            batch_size * num_attention_heads,
            query_length,
            key_length,
            dtype=query.dtype,
            device=key.device,
        )
        if q_quant:
            norm_factor = torch.tensor(1.0, dtype=self.norm_factor.dtype, device=self.norm_factor.device) / self.norm_factor * (1 / q_quant) * (1 / k_quant)
        else:
            norm_factor = torch.tensor(1.0, dtype=self.norm_factor.dtype, device=self.norm_factor.device) / self.norm_factor
        attn_scores = torch.baddbmm(
            attn_scores,
            query,
            key.transpose(1, 2),
            beta=1.0,
            alpha=(norm_factor),
        )
        attn_scores = attn_scores.view(batch_size, num_attention_heads, query_length, key_length)

        mask_value = torch.finfo(attn_scores.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.tensor(mask_value, dtype=attn_scores.dtype).to(attn_scores.device)
        attn_scores = torch.where(causal_mask, attn_scores, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_scores = attn_scores + attention_mask

        attn_weights = nn.functional.softmax(attn_scores, dim=-1)
        attn_weights = attn_weights.to(value.dtype)

        if q_quant:
            attn_weights = (attn_weights * qk_quant).round().clamp(0, 255).to(torch.uint8).to(torch.float32)        
        attn_output = torch.matmul(attn_weights, value)
        if q_quant:
            attn_output = attn_output * ((1 / qk_quant) * (1 / v_quant))            
        return attn_output, attn_weights

class GPTNeoXAttentionExt:
    def __init__(self):
        self.mha = ld.mha_gpt()

    def forward(self, query, key, value, attention_mask, normal_factor, causal_mask = torch.tensor([]), select_nfltmax_at_0 = False):
        return self.mha.exec(query, key, value, torch.tensor([]), attention_mask, causal_mask, select_nfltmax_at_0, normal_factor, False if causal_mask.numel() > 0 else True)

HEAD_NUM = 32
SIZE_PER_HEAD = 80
HIDDEN_SIZE = HEAD_NUM * SIZE_PER_HEAD
MAX_POSITION_EMBEDDINGS = 1024 #2048
def get_ref_model():
    class FakeConfig:
        def __init__(self):
            self.num_attention_heads = HEAD_NUM
            self.hidden_size = HIDDEN_SIZE
            self.max_position_embeddings = MAX_POSITION_EMBEDDINGS
    config = FakeConfig()
    ref_net = GPTNeoXAttention(config)
    ref_net = ref_net.to(dtype=torch.bfloat16)
    return ref_net

def test_gpt_neox():
    inputs = [
        # q, k, v, attn_mask
        # q: [batch, num_heads, query_seq_len, head_size]
        # k: [batch, num_heads, key_seq_len, head_size]
        # v: [batch, num_heads, value_seq_len, head_size]
        # attn: [2, 1, 1, key_seq_len]
        (np.random.random(size=[2, HEAD_NUM, 2, SIZE_PER_HEAD]).astype(np.float32),
         np.random.random(size=[2, HEAD_NUM, 32, SIZE_PER_HEAD]).astype(np.float32),
         np.random.random(size=[2, HEAD_NUM, 32, SIZE_PER_HEAD]).astype(np.float32),
         np.zeros([2, 1, 1, 32], dtype=np.float32)),
        (np.random.random(size=[2, HEAD_NUM, 200, SIZE_PER_HEAD]).astype(np.float32),
         np.random.random(size=[2, HEAD_NUM, 200, SIZE_PER_HEAD]).astype(np.float32),
         np.random.random(size=[2, HEAD_NUM, 200, SIZE_PER_HEAD]).astype(np.float32),
         np.zeros([2, 1, 1, 200], dtype=np.float32)),
        (np.random.random(size=[2, HEAD_NUM, 1, SIZE_PER_HEAD]).astype(np.float32),
         np.random.random(size=[2, HEAD_NUM, 200, SIZE_PER_HEAD]).astype(np.float32),
         np.random.random(size=[2, HEAD_NUM, 200, SIZE_PER_HEAD]).astype(np.float32),
         np.zeros([2, 1, 1, 200], dtype=np.float32)),
    ]
    ref_net = get_ref_model()
    net = GPTNeoXAttentionExt()
    with torch.cpu.amp.autocast():
        for (i, input) in enumerate(inputs):
            q, k, v, attn_mask = input
            q = torch.from_numpy(q).to(torch.bfloat16)
            k = torch.from_numpy(k).to(torch.bfloat16)
            v = torch.from_numpy(v).to(torch.bfloat16)
            attn_mask = torch.from_numpy(attn_mask)
            attn_mask[:,:,:,-2:] = torch.finfo(torch.float32).min
            ref_output = ref_net.forward(q, k, v, attn_mask)
            output = net.forward(q, k, v, attn_mask, normal_factor = 1.0 / math.sqrt(SIZE_PER_HEAD))
            if not torch.allclose(ref_output, output, rtol=0.001, atol=0.01):
                print(f"error at index {i} ref:\n{ref_output} \ncur:\n {output} ")
                assert(False)

    print('done.')
    return

def test_gpt_neox_with_causal():
    inputs = [
        # q, k, v, attn_mask
        # q: [batch, num_heads, query_seq_len, head_size]
        # k: [batch, num_heads, key_seq_len, head_size]
        # v: [batch, num_heads, value_seq_len, head_size]
        # attn: [2, 1, 1, key_seq_len]
        # causal: [2, 1, query_seq_len, key_seq_len]
        # (np.random.random(size=[2, HEAD_NUM, 2, SIZE_PER_HEAD]).astype(np.float32),
        #  np.random.random(size=[2, HEAD_NUM, 32, SIZE_PER_HEAD]).astype(np.float32),
        #  np.random.random(size=[2, HEAD_NUM, 32, SIZE_PER_HEAD]).astype(np.float32),
        #  np.zeros([2, 1, 1, 32], dtype=np.float32)),
        (np.random.random(size=[2, HEAD_NUM, 200, SIZE_PER_HEAD]).astype(np.float32),
         np.random.random(size=[2, HEAD_NUM, 200, SIZE_PER_HEAD]).astype(np.float32),
         np.random.random(size=[2, HEAD_NUM, 200, SIZE_PER_HEAD]).astype(np.float32),
         np.zeros([2, 1, 1, 200], dtype=np.float32)),
        (np.random.random(size=[2, HEAD_NUM, 1, SIZE_PER_HEAD]).astype(np.float32),
         np.random.random(size=[2, HEAD_NUM, 200, SIZE_PER_HEAD]).astype(np.float32),
         np.random.random(size=[2, HEAD_NUM, 200, SIZE_PER_HEAD]).astype(np.float32),
         np.zeros([2, 1, 1, 200], dtype=np.float32)),
    ]
    ref_net = get_ref_model()
    net = GPTNeoXAttentionExt()
    causal_mask = torch.tril(torch.ones((MAX_POSITION_EMBEDDINGS, MAX_POSITION_EMBEDDINGS), dtype=torch.uint8)).view(
                1, 1, MAX_POSITION_EMBEDDINGS, MAX_POSITION_EMBEDDINGS
            )
    with torch.cpu.amp.autocast():
        for (i, input) in enumerate(inputs):
            q, k, v, attn_mask = input
            q = torch.from_numpy(q).to(torch.bfloat16)
            k = torch.from_numpy(k).to(torch.bfloat16)
            v = torch.from_numpy(v).to(torch.bfloat16)
            
            batch_size, num_attention_heads, query_length, attn_head_size = q.size()
            key_length = k.size(-2)
            causal_mask_sub = causal_mask[:, :, key_length - query_length : key_length, :key_length].contiguous()
            # 0 means -fltmax
            select_nfltmax_at_0 = True
            if i == 0:
                # 1 means -fltmax
                causal_mask_sub = 1 - causal_mask_sub
                select_nfltmax_at_0 = False
            attn_mask = torch.from_numpy(attn_mask)
            attn_mask[:,:,:,-2:] = torch.finfo(torch.float32).min
            ref_output = ref_net.forward(q, k, v, attn_mask)
            output = net.forward(q, k, v, attn_mask, normal_factor = 1.0 / math.sqrt(SIZE_PER_HEAD), causal_mask = causal_mask_sub, select_nfltmax_at_0 = select_nfltmax_at_0)
            if not torch.allclose(ref_output, output, rtol=0.001, atol=0.01):
                print(f"error at index {i} ref:\n{ref_output} \ncur:\n {output} ")
                assert(False)

    print('done.')
    return

if __name__ == "__main__":
    test_gpt_neox_with_causal()
