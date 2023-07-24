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

# copy from transformers/models/bloom/modeling_bloom.py
class BloomAttention(nn.Module):
    def __init__(self, head_dim:int, num_heads:int):
        super().__init__()

        # self.pretraining_tp = config.pretraining_tp
        # self.slow_but_exact = config.slow_but_exact

        # self.hidden_size = config.hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        # self.split_size = self.hidden_size
        # self.hidden_dropout = config.hidden_dropout

        # if self.head_dim * self.num_heads != self.hidden_size:
        #     raise ValueError(
        #         f"`hidden_size` must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`:"
        #         f" {self.num_heads})."
        #     )

        # Layer-wise attention scaling
        self.inv_norm_factor = 1.0 / math.sqrt(head_dim)
        self.beta = 1.0

        # self.query_key_value = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=True)
        # self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        # self.attention_dropout = nn.Dropout(config.attention_dropout)

    # def _split_heads(self, fused_qkv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     """
    #     Split the last dimension into (num_heads, head_dim) without making any copies, results share same memory
    #     storage as `fused_qkv`

    #     Args:
    #         fused_qkv (`torch.tensor`, *required*): [batch_size, seq_length, num_heads * 3 * head_dim]

    #     Returns:
    #         query: [batch_size, seq_length, num_heads, head_dim] key: [batch_size, seq_length, num_heads, head_dim]
    #         value: [batch_size, seq_length, num_heads, head_dim]
    #     """
    #     batch_size, seq_length, three_times_hidden_size = fused_qkv.shape
    #     fused_qkv = fused_qkv.view(batch_size, seq_length, self.num_heads, 3, self.head_dim)
    #     return fused_qkv[..., 0, :], fused_qkv[..., 1, :], fused_qkv[..., 2, :]

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Merge heads together over the last dimenstion

        Args:
            x: (`torch.tensor`, *required*): [batch_size * num_heads, seq_length, head_dim]

        Returns:
            torch.tensor: [batch_size, seq_length, num_heads * head_dim]
        """
        # What we want to achieve is:
        # batch_size * num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads * head_dim
        batch_size_and_num_heads, seq_length, _ = x.shape
        batch_size = batch_size_and_num_heads // self.num_heads

        # First view to decompose the batch size
        # batch_size * num_heads, seq_length, head_dim -> batch_size, num_heads, seq_length, head_dim
        x = x.view(batch_size, self.num_heads, seq_length, self.head_dim)

        # batch_size, num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads, head_dim
        x = x.permute(0, 2, 1, 3)

        # batch_size, seq_length, num_heads, head_dim -> batch_size, seq_length, num_heads * head_dim
        return x.reshape(batch_size, seq_length, self.num_heads * self.head_dim)

    def forward(
        self,
        query_layer: torch.Tensor,    # [batch * head_num, q_len, head_size]
        key_layer: torch.Tensor,      # [batch * head_num, head_size, q_len+kv_len]
        value_layer: torch.Tensor,    # [batch * head_num, q_len+kv_len, head_size]
        alibi: torch.Tensor,          # [batch * head_num, 1, q_len+kv_len]
        attention_mask: torch.Tensor, # [batch * head_num, q_len, q_len+kv_len]
    ):
        # fused_qkv = self.query_key_value(hidden_states)  # [batch_size, seq_length, 3 x hidden_size]

        # # 3 x [batch_size, seq_length, num_heads, head_dim]
        # (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)

        batch_size, _, q_length, _ = query_layer.shape

        query_layer = query_layer.reshape(-1, q_length, self.head_dim)
        key_layer = key_layer.reshape(-1, key_layer.size(2), key_layer.size(3))
        value_layer = value_layer.reshape(-1, value_layer.size(2), value_layer.size(3))
        # if layer_past is not None:
        #     past_key, past_value = layer_past
        #     # concatenate along seq_length dimension:
        #     #  - key: [batch_size * self.num_heads, head_dim, kv_length]
        #     #  - value: [batch_size * self.num_heads, kv_length, head_dim]
        #     key_layer = torch.cat((past_key, key_layer), dim=2)
        #     value_layer = torch.cat((past_value, value_layer), dim=1)

        _, _, kv_length = key_layer.shape

        # if use_cache is True:
        #     present = (key_layer, value_layer)
        # else:
        #     present = None

        # [batch_size * num_heads, q_length, kv_length]
        # we use `torch.Tensor.baddbmm` instead of `torch.baddbmm` as the latter isn't supported by TorchScript v1.11
        matmul_result = alibi.baddbmm(
            batch1=query_layer,
            batch2=key_layer,
            beta=self.beta,
            alpha=self.inv_norm_factor,
        )

        # change view to [batch_size, num_heads, q_length, kv_length]
        attention_scores = matmul_result.view(batch_size, self.num_heads, q_length, kv_length)

        # cast attention scores to fp32, compute scaled softmax and cast back to initial dtype - [batch_size, num_heads, q_length, kv_length]
        input_dtype = attention_scores.dtype
        # `float16` has a minimum value of -65504.0, whereas `bfloat16` and `float32` have a minimum value of `-3.4e+38`
        if input_dtype == torch.float16:
            attention_scores = attention_scores.to(torch.float)
        #attn_weights = torch.masked_fill(attention_scores, attention_mask, torch.finfo(attention_scores.dtype).min)
        attn_weights = attention_scores + attention_mask
        attention_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(input_dtype)

        # change view [batch_size x num_heads, q_length, kv_length]
        attention_probs_reshaped = attention_probs.view(batch_size * self.num_heads, q_length, kv_length)

        # matmul: [batch_size * num_heads, q_length, head_dim]
        context_layer = torch.bmm(attention_probs_reshaped, value_layer)

        # change view [batch_size, num_heads, q_length, head_dim]
        context_layer = self._merge_heads(context_layer)

        # aggregate results across tp ranks. See here: https://github.com/pytorch/pytorch/issues/76232
        # if self.pretraining_tp > 1 and self.slow_but_exact:
        #     slices = self.hidden_size / self.pretraining_tp
        #     output_tensor = torch.zeros_like(context_layer)
        #     for i in range(self.pretraining_tp):
        #         output_tensor = output_tensor + F.linear(
        #             context_layer[:, :, int(i * slices) : int((i + 1) * slices)],
        #             self.dense.weight[:, int(i * slices) : int((i + 1) * slices)],
        #         )
        # else:
        #     output_tensor = self.dense(context_layer)

        # output_tensor = dropout_add(output_tensor, residual, self.hidden_dropout, self.training)

        # outputs = (output_tensor, present)
        # if output_attentions:
        #     outputs += (attention_probs,)

        return context_layer

class BloomAttentionExt:
    def __init__(self):
        self.mha = ld.mha_gpt()

    def forward(self, query, key, value, alibi, attention_mask, normal_factor):
        return self.mha.exec(query, key, value, alibi, attention_mask, torch.tensor([]), False, normal_factor, False)

HEAD_NUM = 32
SIZE_PER_HEAD = 80
HIDDEN_SIZE = HEAD_NUM * SIZE_PER_HEAD
MAX_POSITION_EMBEDDINGS = 1024 #2048
def get_ref_model():
    ref_net = BloomAttention(SIZE_PER_HEAD, HEAD_NUM)
    ref_net = ref_net.to(dtype=torch.bfloat16)
    return ref_net

def test_bloom():
    inputs = [
        # q, k, v, attn_mask
        # q: [batch, num_heads, query_seq_len, head_size]
        # k: [batch, num_heads, head_size, key_seq_len]
        # v: [batch, num_heads, value_seq_len, head_size]
        # alibi: [batch, num_heads, 1, key_seq_len]
        # attn: [2, 1, query_seq_len, key_seq_len]
        (np.random.random(size=[2, HEAD_NUM, 2, SIZE_PER_HEAD]).astype(np.float32),
         np.random.random(size=[2, HEAD_NUM, SIZE_PER_HEAD, 32]).astype(np.float32),
         np.random.random(size=[2, HEAD_NUM, 32, SIZE_PER_HEAD]).astype(np.float32),
         np.random.random(size=[2, HEAD_NUM, 1, 32]).astype(np.float32),
         np.zeros([2, 1, 2, 32], dtype=np.float32)),
        (np.random.random(size=[2, HEAD_NUM, 200, SIZE_PER_HEAD]).astype(np.float32),
         np.random.random(size=[2, HEAD_NUM, SIZE_PER_HEAD, 200]).astype(np.float32),
         np.random.random(size=[2, HEAD_NUM, 200, SIZE_PER_HEAD]).astype(np.float32),
         np.random.random(size=[2, HEAD_NUM, 1, 200]).astype(np.float32),
         np.zeros([2, 1, 200, 200], dtype=np.float32)),
        (np.random.random(size=[2, HEAD_NUM, 1, SIZE_PER_HEAD]).astype(np.float32),
         np.random.random(size=[2, HEAD_NUM, SIZE_PER_HEAD, 200]).astype(np.float32),
         np.random.random(size=[2, HEAD_NUM, 200, SIZE_PER_HEAD]).astype(np.float32),
         np.random.random(size=[2, HEAD_NUM, 1, 200]).astype(np.float32),
         np.zeros([2, 1, 1, 200], dtype=np.float32)),
    ]
    ref_net = get_ref_model()
    net = BloomAttentionExt()
    with torch.cpu.amp.autocast():
        for (i, input) in enumerate(inputs):
            q, k, v, alibi, attn_mask = input
            q = torch.from_numpy(q).to(torch.bfloat16)
            k = torch.from_numpy(k).to(torch.bfloat16)
            v = torch.from_numpy(v).to(torch.bfloat16)
            alibi = torch.from_numpy(alibi) # to(torch.bfloat16)
            attn_mask = torch.from_numpy(attn_mask)
            attn_mask[:,:,:,-2:] = torch.finfo(torch.float32).min
            output = net.forward(q, k, v, alibi, attn_mask, normal_factor = 1.0 / math.sqrt(SIZE_PER_HEAD))
            alibi = alibi.view(-1, alibi.size(2), alibi.size(3))
            ref_output = ref_net.forward(q, k, v, alibi, attn_mask)
            if not torch.allclose(ref_output, output, rtol=0.001, atol=0.01):
                print(f"error at index {i} ref:\n{ref_output} \ncur:\n {output} ")
                assert(False)

    print('done.')
    return

if __name__ == "__main__":
    test_bloom()
