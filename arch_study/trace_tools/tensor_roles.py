#!/usr/bin/env python3
"""
tensor_roles.py - names and roles of the tensors each AI kernel allocates,
in allocation order (read from ai_kernels/src/*.c; those files are not
changed). The "map" binaries print one TENSORMAP line per allocation in the
same order, so entry i of the list below names allocation i.

Roles (the tensor-type feature used by the tensor-aware policy):
  weight          parameters read every inference (GEMM B, conv filters, LSTM W)
  activation_in   layer input produced by the previous layer
  activation_out  layer output consumed by the next layer
  kv_cache        attention K/V cache (decode)
  gather_table    table read at data-dependent rows (embeddings, graph features)
  index           integer indices that drive a gather
  scratch         small temporaries reused inside the layer
  activation_mid  (chains only) output of layer k that is the input of layer k+1
  flush           the 64 MiB cold-start buffer (freed before the ROI)

For the layer chains (src/chain.c) every tensor also lists the layers (1-based,
in execution order) that use it: the basis of the "next-layer hint" and
"dead tensor" features. Layer k is the function layer_<prefix>_<k>_<name>.
"""

W, AI, AO, KV, GT, IX, SC = ("weight", "activation_in", "activation_out",
                             "kv_cache", "gather_table", "index", "scratch")
MID = "activation_mid"

# chain -> (function prefix, layer names, [(tensor, role, layers using it)])
_TF = [("O", AI, [1]), ("Wo", W, [1]), ("X1", MID, [1, 2]), ("W1", W, [2]),
       ("H", MID, [2, 3]), ("W2", W, [3]), ("Y", AO, [3])]
_CV = [("X", AI, [1]), ("Wa", W, [1]), ("A", MID, [1, 2]), ("Wb", W, [2]),
       ("B", MID, [2, 3]), ("Wc", W, [3]), ("Y", AO, [3])]
_RN = [("x", AI, [1])]
for _k, _out in ((1, MID), (2, MID), (3, AO)):
    _RN += [(f"W_ih{_k}", W, [_k]), (f"W_hh{_k}", W, [_k]), (f"b{_k}", W, [_k]),
            (f"y{_k}", _out, [_k, _k + 1] if _k < 3 else [_k]),
            (f"h{_k}", SC, [_k]), (f"c{_k}", SC, [_k]), (f"z{_k}", SC, [_k])]
# whole blocks (src/block.c, Step 2): block -> (function prefix, layer names, tensors)
_EB = [("X", AI, [1, 4]), ("Wq", W, [1]), ("Wk", W, [1]), ("Wv", W, [1]),
       ("Q", MID, [1, 2]), ("K", MID, [1, 2]), ("V", MID, [1, 2]), ("P", SC, [2]),
       ("C", MID, [2, 3]), ("Wo", W, [3]), ("A", MID, [3, 4]), ("X1", MID, [4, 5, 7]),
       ("W1", W, [5]), ("H", MID, [5, 6]), ("W2", W, [6]), ("Y", MID, [6, 7]),
       ("Out", AO, [7])]
_EB_LAYERS = ["qkv", "attn", "oproj", "add_ln1", "ffn_up", "ffn_down", "add_ln2"]
_CB = [("X", AI, [1, 3]), ("Wa1", W, [1]), ("A1", MID, [1, 2]), ("Wb1", W, [2]),
       ("B1", MID, [2, 3]), ("Wc1", W, [3]), ("Y1", MID, [3, 4, 6]), ("Wa2", W, [4]),
       ("A2", MID, [4, 5]), ("Wb2", W, [5]), ("B2", MID, [5, 6]), ("Wc2", W, [6]),
       ("Y2", AO, [6])]
_CB_LAYERS = ["b1_expand", "b1_conv3x3", "b1_project", "b2_expand", "b2_conv3x3", "b2_project"]
_DL = [("dense", AI, [1])]
for _l, _r in ((0, SC), (1, SC), (2, MID)):
    _DL += [(f"bot_W{_l}", W, [1]), (f"bot_a{_l}", _r, [1, 3] if _r == MID else [1])]
_DL += [(f"table{_t}", GT, [2]) for _t in range(26)]
_DL += [("idx", IX, [2]), ("pool", MID, [2, 3]), ("z", MID, [3, 4])]
for _l in range(5):
    _DL += [(f"top_W{_l}", W, [4]), (f"top_a{_l}", AO if _l == 4 else SC, [4])]
BLOCKS = {
    "bert_block": ("eb", _EB_LAYERS, _EB),
    "vit_block": ("eb", _EB_LAYERS, _EB),
    "rn50_block2": ("cb", _CB_LAYERS, _CB),
    "mbv2_block2": ("cb", _CB_LAYERS, _CB),
    "dlrm_block": ("dl", ["bot_mlp", "embbag", "interact", "top_mlp"], _DL),
    # Step 8 unseen workloads (ai_kernels/src/block_unseen.c, same code paths)
    "bertbase_block": ("eb", _EB_LAYERS, _EB),
    "gpt2xl_block": ("eb", _EB_LAYERS, _EB),
    "dlrm_b128": ("dl", ["bot_mlp", "embbag", "interact", "top_mlp"], _DL),
    "rn50_block3": ("cb", _CB_LAYERS, _CB),
}

_LL = [("x", AI, [1, 2]), ("Wq", W, [1]), ("q", MID, [1, 3]), ("Wk", W, [2]),
       ("Wv", W, [2]), ("k_new", SC, [2]), ("v_new", SC, [2]), ("K_cache", KV, [2, 3]),
       ("V_cache", KV, [2, 3]), ("p", SC, [3]), ("o", MID, [3, 4]), ("Wo", W, [4]),
       ("y", MID, [4, 5]), ("W_up", W, [5]), ("h", MID, [5, 6]), ("W_down", W, [6]),
       ("out", AO, [6])]
CHAINS = {
    "bert_ffn": ("tf", ["oproj", "ffn_up", "ffn_down"], _TF),
    "vit_mlp": ("tf", ["oproj", "ffn_up", "ffn_down"], _TF),
    "rn50_bottleneck": ("cv", ["expand", "conv3x3", "project"], _CV),
    "mbv2_block": ("cv", ["expand", "conv3x3", "project"], _CV),
    "rnnt_encoder": ("rn", ["pre1", "pre2", "post1"], _RN),
    "llama_decode": ("ll", ["q_proj", "kv_proj", "attn", "o_proj", "ffn_up", "ffn_down"], _LL),
}


def tensor_users(kernel, args):
    """Layers using each tensor (chains only), aligned with tensor_list()."""
    if kernel == "block":
        return [u for _, _, u in BLOCKS[args.split()[0].split(":")[0]][2]]
    if kernel != "chain":
        return [[1] for _ in tensor_list(kernel, args)]
    return [u for _, _, u in CHAINS[args.split()[0]][2]]


def tensor_list(kernel, args):
    """Return [(name, role), ...] in allocation order (without the flush buffer)."""
    a = args.split()
    if kernel == "chain":
        return [(n, r) for n, r, _ in CHAINS[a[0]][2]]
    if kernel == "block":
        return [(n, r) for n, r, _ in BLOCKS[a[0].split(":")[0]][2]]
    if kernel == "gemm":
        return [("A", AI), ("B", W), ("C", AO)]
    if kernel in ("conv2d_nchw", "conv3d", "depthwise_conv"):
        return [("in", AI), ("wt", W), ("out", AO)]
    if kernel == "pool":
        return [("in", AI), ("out", AO)]
    if kernel == "attention":
        return [("Q", AI), ("Kt", AI), ("V", AI), ("O", AO), ("P", SC)]
    if kernel == "attention_decode":
        return [("q", AI), ("K", KV), ("V", KV), ("o", AO), ("p", SC)]
    if kernel == "lstm":
        return [("x", AI), ("W_ih", W), ("W_hh", W), ("bias", W), ("y", AO),
                ("h", SC), ("c", SC), ("z", SC)]
    if kernel == "gat_aggregate":
        return [("feat", GT), ("a_src", W), ("a_dst", W), ("out", AO), ("e", SC),
                ("nbr", IX)]
    if kernel == "embedding":
        if a[0] == "gather":
            return [("table", GT), ("out", AO), ("idx", IX)]
        return ([(f"table{t}", GT) for t in range(26)]
                + [("idx", IX), ("out", AO)])
    if kernel == "mlp":
        nl = len(a[0].split(",")) - 1
        lst = []
        for l in range(nl):
            lst += [(f"W{l}", W), (f"b{l}", W)]
        return lst + [("X", AI), ("out", AO), ("act0", SC), ("act1", SC)]
    raise ValueError(f"unknown kernel {kernel}")


def label(kernel, args, n_allocs):
    """Names/roles for n_allocs recorded allocations; extra trailing ones
    (the cold-mode flush buffer) are labelled 'flush'."""
    lst = tensor_list(kernel, args)
    if n_allocs < len(lst):
        raise ValueError(f"{kernel}: {n_allocs} allocations recorded, "
                         f"{len(lst)} expected")
    return lst + [("flush", "flush")] * (n_allocs - len(lst))
