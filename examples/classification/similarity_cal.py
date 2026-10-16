# similarity_cal.py
import os
import torch

def _unwrap_model(m):
    return m.module if hasattr(m, "module") else m

def _find_order_prompt_encoder(model_like):
    m = _unwrap_model(model_like)

    if hasattr(m, "encoder"):
        enc = getattr(m, "encoder")
        if hasattr(enc, "use_order_prompt") and getattr(enc, "use_order_prompt", False) \
           and hasattr(enc, "order_prompt") \
           and hasattr(enc, "per_layer_prompt_indexe") \
           and hasattr(enc, "mamba_layers_orders") \
           and hasattr(enc, "order_prompt_proj") \
           and hasattr(enc, "mamba_blocks_list"):
            return enc

        for mod in enc.modules():
            if hasattr(mod, "use_order_prompt") and getattr(mod, "use_order_prompt", False) \
               and hasattr(mod, "order_prompt") \
               and hasattr(mod, "per_layer_prompt_indexe") \
               and hasattr(mod, "mamba_layers_orders") \
               and hasattr(mod, "order_prompt_proj") \
               and hasattr(mod, "mamba_blocks_list"):
                return mod

    for mod in m.modules():
        if hasattr(mod, "use_order_prompt") and getattr(mod, "use_order_prompt", False) \
           and hasattr(mod, "order_prompt") \
           and hasattr(mod, "per_layer_prompt_indexe") \
           and hasattr(mod, "mamba_layers_orders") \
           and hasattr(mod, "order_prompt_proj") \
           and hasattr(mod, "mamba_blocks_list"):
            return mod

    return None


@torch.no_grad()
def compute_order_similarity_rankings(model_like, order_vocab):
    enc = _find_order_prompt_encoder(model_like)
    if enc is None:
        raise RuntimeError("Cannot find encoder with order_prompt.")

    weights = enc.order_prompt.weight

    layer2stage = []
    for s in range(len(enc.mamba_blocks_list)):
        n_layers_s = len(enc.mamba_blocks_list[s])
        layer2stage += [s] * n_layers_s

    if len(layer2stage) != len(enc.mamba_layers_orders):
        raise RuntimeError("Stage/layer config mismatch.")

    stage_order_to_g = {}
    for li, order in enumerate(enc.mamba_layers_orders):
        if order not in order_vocab:
            continue

        s = layer2stage[li]
        start, end = enc.per_layer_prompt_indexe[li]
        e_bar = weights[start:end].mean(dim=0)

        v = enc.order_prompt_proj[s](e_bar.unsqueeze(0)).squeeze(0)
        W = enc.order_prompt_proj[s].weight
        g = torch.matmul(W.t(), v)

        stage_order_to_g.setdefault((s, order), []).append(g)

    stage_order_g = {}
    for (s, order), g_list in stage_order_to_g.items():
        stage_order_g[(s, order)] = torch.stack(g_list, dim=0).mean(dim=0)

    stage_order_g_hat = {}
    for (s, order), g in stage_order_g.items():
        stage_order_g_hat[(s, order)] = g / (g.norm(p=2) + 1e-12)

    order_to_vecs = {o: [] for o in order_vocab}
    for (s, o), gh in stage_order_g_hat.items():
        if o in order_to_vecs:
            order_to_vecs[o].append(gh)

    final_proto = []
    for o in order_vocab:
        vecs = order_to_vecs[o]
        if len(vecs) == 0:
            raise RuntimeError(f"Order '{o}' not found in encoder.mamba_layers_orders.")
        v = torch.stack(vecs, dim=0).mean(dim=0)
        v = v / (v.norm(p=2) + 1e-12)
        final_proto.append(v)

    protos = torch.stack(final_proto, dim=0)
    sim = (protos @ protos.t()).float().cpu().numpy()

    rankings = {}
    for i, oi in enumerate(order_vocab):
        pairs = []
        for j, oj in enumerate(order_vocab):
            if i == j:
                continue
            pairs.append((oj, float(sim[i, j])))
        pairs.sort(key=lambda x: x[1], reverse=True)
        rankings[oi] = pairs

    return rankings, sim


def log_order_similarity(model_like, order_vocab, save_path, epoch=None):
    rankings, sim = compute_order_similarity_rankings(model_like, order_vocab)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "a", encoding="utf-8") as f:
        if epoch is not None:
            f.write(f"Epoch {epoch:03d}\n")
        for oi in order_vocab:
            pairs = rankings[oi]
            f.write(oi + ": " + ", ".join([f"{oj}({s:.4f})" for oj, s in pairs]) + "\n")
        f.write("-" * 60 + "\n")

    return rankings, sim