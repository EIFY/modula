import jax.numpy as jnp

from modula.abstract import *
from modula.atom import *
from modula.bond import *

def MLP(output_dim, input_dim, width, depth):
    m = Linear(output_dim, width) @ ReLU()
    for _ in range(depth-2):
        m = m @ Linear(width, width) @ ReLU()
    return m @ Linear(width, input_dim)

def Attention(num_heads, d_embed, d_query, d_value, softmax_scale, causal, posemb="rope", bias=False):
    """Multi-head attention"""
    Q, K, V = Linear(num_heads * d_query, d_embed), Linear(num_heads * d_query, d_embed), Linear(num_heads * d_value, d_embed)
    Q = SplitIntoHeads(num_heads) @ (Q + Bias(num_heads * d_query) if bias else Q)
    K = SplitIntoHeads(num_heads) @ (K + Bias(num_heads * d_query) if bias else K)
    V = SplitIntoHeads(num_heads) @ (V + Bias(num_heads * d_value) if bias else V)
    W = Linear(d_embed, num_heads * d_value)
    W = (W + Bias(d_embed) if bias else W) @ MergeHeads()
    QK = (Q, K)
    if posemb == "rope":
        QK = Rope(d_query) @ QK
    attn = AttentionQK() @ QK
    if causal:
        attn = CausalMask() @ attn
    AttentionScores = Softmax(softmax_scale) @ attn
    return W @ (1/3 * ApplyAttentionScores()) @ (V, AttentionScores)

def GPT(vocab_size, num_heads, d_embed, d_query, d_value, num_blocks, blocks_mass=5, attention_scale=1.0, final_scale=1.0):
    embed = Embed(d_embed, vocab_size)
    embed.tare()

    att = Attention(num_heads, d_embed, d_query, d_value, attention_scale, causal=True)
    mlp = Linear(d_embed, 4*d_embed) @ GeLU() @ Linear(4*d_embed, d_embed)
    att_block = (1-1/(2*num_blocks)) * Identity() + 1/(2*num_blocks) * att
    mlp_block = (1-1/(2*num_blocks)) * Identity() + 1/(2*num_blocks) * mlp
    blocks = (mlp_block @ att_block) ** num_blocks
    blocks.tare(absolute=blocks_mass)

    out = final_scale * Linear(vocab_size, d_embed)

    return out @ blocks @ embed

def posemb_sincos_2d(h, w, width, temperature=10_000., dtype=jnp.float32):
    """Follows the MoCo v3 logic."""
    y, x = jnp.mgrid[:h, :w]

    assert width % 4 == 0, "Width must be mult of 4 for sincos posemb"
    omega = jnp.arange(width // 4) / (width // 4 - 1)
    omega = 1. / (temperature**omega)
    y = jnp.einsum("m,d->md", y.flatten(), omega)
    x = jnp.einsum("m,d->md", x.flatten(), omega)
    pe = jnp.concatenate([jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=1)
    return jnp.asarray(pe, dtype)[None, :, :]

def ViT(num_classes, image_size=(28, 28), patch_size=(7, 7), num_heads=4, d_embed=32, d_query=8, d_value=8, num_blocks=4, blocks_mass=5, attention_scale=1.0, final_scale=1.0, channels=1, LN=True, bias=True, scale=True):
    i1, i2 = image_size
    p1, p2 = patch_size
    h, w = i1 // p1, i2 // p2
    patchify = Linear(d_embed, p1 * p2 * channels) @ Patchify(patch_size)
    if bias:
        patchify = patchify + Bias(d_embed)
    patchify.name = 'embedding'
    posemb = Constant(lambda: posemb_sincos_2d(h, w, d_embed))

    att = Attention(num_heads, d_embed, d_query, d_value, attention_scale, causal=False, posemb="none", bias=bias)
    mlp = (Linear(d_embed, 4*d_embed) + Bias(d_embed) if bias else Linear(d_embed, 4*d_embed)) @ GeLU() @ (Linear(4*d_embed, d_embed) + Bias(4*d_embed) if bias else Linear(4*d_embed, d_embed))
    if LN:
        ln = LayerNorm()
        if bias and scale:
            ln = (Scale(d_embed) + Bias(d_embed)) @ ln
        elif bias:
            ln = ln + Bias(d_embed)
        elif scale:
            ln = Scale(d_embed) @ ln
        att = att @ ln
        mlp = mlp @ ln
    att_block = (1-1/(2*num_blocks)) * Identity() + 1/(2*num_blocks) * att
    att_block.name = 'att_block'
    mlp_block = (1-1/(2*num_blocks)) * Identity() + 1/(2*num_blocks) * mlp
    mlp_block.name = 'mlp_block'
    encoder_block = mlp_block @ att_block
    encoder_block.name = 'encoder_block'
    blocks = encoder_block ** num_blocks
    blocks.tare(absolute=blocks_mass)

    gap = Mean(axis=1, size=h * w)
    out = final_scale * (Linear(num_classes, d_embed) + Bias(num_classes) if bias else Linear(num_classes, d_embed))
    out.name = 'head'

    ret = blocks @ (patchify + posemb)
    if LN:  # Final LN
        ret = ln @ ret
    return out @ gap @ ret

def extract_target_norm(vit):
    extract = lambda a: (type(a).__name__, a.target_norm)
    ret = {}
    def extract_add(m):
        t, add = m.children
        return t.children
    def extract_ln(prefix, ln):
        ln, scale_bias = ln.children
        scale, bias = extract_add(scale_bias)
        ret[prefix + 'bias'] = extract(bias)
        ret[prefix + 'scale'] = extract(scale)
    def extract_kernel_bias(prefix, m):
        kernel, bias = extract_add(m)
        ret[prefix + 'bias'] = extract(bias)
        ret[prefix + 'kernel'] = extract(kernel)
    def extract_att(prefix, att):
        v_score, apply_att_out = att.children
        apply_att, w = apply_att_out.children
        merge_head, w = w.children
        extract_kernel_bias(prefix + 'out/', w)
        v, score = v_score.children
        v, split_head = v.children
        extract_kernel_bias(prefix + 'value/', v)
        attn, softmax = score.children
        qk, attn_qk = attn.children
        q, k = qk.children
        q, split_head = q.children
        extract_kernel_bias(prefix + 'query/', q)
        k, split_head = k.children
        extract_kernel_bias(prefix + 'key/', k)
    def extract_mlp(prefix, mlp):
        mlp0, gelu_mlp1 = mlp.children
        extract_kernel_bias(prefix + 'Dense_0/', mlp0)
        gelu, mlp1 = gelu_mlp1.children
        extract_kernel_bias(prefix + 'Dense_1/', mlp1)
    def extract_encoder_block(b):
        l = b.name.split('_')
        n = int(l[-1])
        att_b, mlp_b = b.children
        res, att = extract_add(att_b)
        att, mul = att.children
        ln, att = att.children
        extract_ln(f"Transformer/encoderblock_{n}/LayerNorm_0/", ln)
        extract_att(f"Transformer/encoderblock_{n}/MultiHeadDotProductAttention_0/", att)
        res, mlp = extract_add(mlp_b)
        mlp, mul = mlp.children
        ln, mlp = mlp.children
        extract_ln(f"Transformer/encoderblock_{n}/LayerNorm_1/", ln)
        extract_mlp(f"Transformer/encoderblock_{n}/MlpBlock_0/", mlp)
    headless, pool_head = vit.children
    pool, head = pool_head.children
    assert head.name == 'head'
    head, mul = head.children
    head_tuple, add = head.children
    kernel, bias = head_tuple.children
    ret['head/bias'] = extract(bias)
    ret['head/kernel'] = extract(kernel)
    headless, final_ln = headless.children
    extract_ln('Transformer/encoder_norm/', final_ln)
    embedding, transformer = headless.children
    patchify, posemb = extract_add(embedding)
    assert patchify.name == 'embedding'
    patchify, bias = extract_add(patchify)
    ret['embedding/bias'] = extract(bias)
    rearrange, patchify = patchify.children
    ret['embedding/kernel'] = extract(patchify)
    top = transformer
    while type(top) is CompositeModule:
        top, last = top.children
        extract_encoder_block(last)
    return ret
