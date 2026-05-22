import torch
import xformers.ops

# From https://github.com/facebookresearch/xformers/blob/042abc8aa47d1f5bcc2e82df041811de218924ba/tests/test_mem_eff_attention.py#L511 # noqa


def ref_attention(q, k, v, attn_bias=None, drop_mask=None, p=0.0, scale=None):
    q = q.float()
    k = k.float()
    v = v.float()

    scale = scale if scale is not None else (1 / q.shape[-1] ** 0.5)
    q = q * scale

    attn = q @ k.transpose(-2, -1)
    if attn_bias is not None:
        if isinstance(attn_bias, xformers.ops.AttentionBias):
            # Always create in B,H,Mq,Mk format
            attn_bias_tensor = attn_bias.materialize(
                (q.shape[0], 1, q.shape[1], k.shape[1]),
                device=q.device,
                dtype=torch.float32,
            )
        else:
            attn_bias_tensor = attn_bias
        if attn_bias_tensor.ndim == 4:
            assert q.shape[0] == attn_bias_tensor.shape[0] * attn_bias_tensor.shape[1]
            attn_bias_tensor = attn_bias_tensor.reshape([-1, *attn_bias_tensor.shape[2:]])
        attn = attn + attn_bias_tensor.float()
    attn = attn.softmax(-1)
    if drop_mask is not None:
        attn = attn * (drop_mask / (1 - p))
    return attn @ v


def ref_attention_bmhk(q, k, v, attn_bias, scale=None) -> torch.Tensor:
    assert q.ndim == 4

    def T(t):
        return t.permute((0, 2, 1, 3)).reshape([t.shape[0] * t.shape[2], t.shape[1], t.shape[3]])

    if isinstance(attn_bias, xformers.ops.AttentionBias):
        attn_bias = attn_bias.materialize(
            (q.shape[0], q.shape[2], q.shape[1], k.shape[1]),
            device=q.device,
            dtype=torch.float32,
        ).reshape([q.shape[0] * q.shape[2], q.shape[1], k.shape[1]])
    out = ref_attention(T(q), T(k), T(v), attn_bias, scale=scale)
    out = out.reshape([q.shape[0], q.shape[2], q.shape[1], v.shape[3]])
    return out.permute((0, 2, 1, 3))


def memory_efficient_attention_wrapper(q, k, v, attn_bias):
    if q.device.type == "cpu":
        return ref_attention_bmhk(q, k, v, attn_bias)
    else:
        return xformers.ops.memory_efficient_attention(q, k, v, attn_bias)


def attention_with_weights_wrapper(q, k, v, attn_bias):
    """Compute attention and return (output [B,Mq,H,K], weights [B,H,Mq,Mk]).

    Always materializes the full attention matrix, so use only at inference
    time when you need the weights (e.g. for attention rollout).

    Memory layout: keep q/k/v and the bias in the input dtype (bf16/fp16 under
    autocast) and run the softmax in that same dtype, in place on `scores`.
    This roughly halves peak memory for the [Mq, Mk] tensors versus
    materializing everything in fp32, and avoids a second [Mq, Mk] buffer for
    the softmax output. The precision tradeoff matches what the efficient
    attention kernels we'd otherwise use already accept.
    """
    assert q.ndim == 4  # BMHK
    B, Mq, H, K = q.shape
    Mk = k.shape[1]

    def T(t):
        return t.permute((0, 2, 1, 3)).reshape([t.shape[0] * t.shape[2], t.shape[1], t.shape[3]])

    # Reduced-precision dtype for memory-heavy tensors; fp32 fallback if the
    # caller is already in fp32.
    compute_dtype = q.dtype if q.dtype in (torch.bfloat16, torch.float16) else torch.float32

    if isinstance(attn_bias, xformers.ops.AttentionBias):
        attn_bias_mat = attn_bias.materialize(
            (B, H, Mq, Mk), device=q.device, dtype=compute_dtype
        ).reshape([B * H, Mq, Mk])
    elif attn_bias is not None:
        attn_bias_mat = attn_bias.to(compute_dtype)
    else:
        attn_bias_mat = None

    q_c, k_c, v_c = T(q).to(compute_dtype), T(k).to(compute_dtype), T(v).to(compute_dtype)
    scale = 1.0 / q_c.shape[-1] ** 0.5
    scores = (q_c * scale) @ k_c.transpose(-2, -1)   # [B*H, Mq, Mk]
    if attn_bias_mat is not None:
        scores += attn_bias_mat
    del attn_bias_mat
    # Numerically stable softmax done in-place on `scores` so we don't allocate
    # a second [Mq, Mk] tensor for the result. (torch.softmax has no in-place
    # variant; the efficient attention kernels we'd otherwise use likewise
    # operate in the compute dtype, so the precision tradeoff is the same.)
    scores -= scores.amax(dim=-1, keepdim=True)
    torch.exp_(scores)
    scores /= scores.sum(dim=-1, keepdim=True)
    weights = scores                                          # [B*H, Mq, Mk]
    out = (weights @ v_c).to(q.dtype)                         # [B*H, Mq, K]

    out = out.reshape([B, H, Mq, K]).permute((0, 2, 1, 3))   # [B, Mq, H, K]
    weights = weights.reshape([B, H, Mq, Mk])                 # [B, H, Mq, Mk]
    return out, weights
