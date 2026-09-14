"""Vectorized replacement for `node_update.forward(training=True)`.

The baseline forward pass loops over samples, and inside every message-passing
layer it loops over AP-UE nodes, so one training iteration builds an autograd
graph out of thousands of tiny ops. Profiling a batch-8 iteration gives 4.7% in
data generation and 95% in forward/backward, which caps the usable training
budget well below convergence.

This module recomputes the same mathematics in batched form. It defines no
parameters of its own: it reads the weights out of the existing `node_update`
instance, so the same checkpoints load unchanged and `model_2.py` is untouched.
`verify_equivalence` checks outputs and parameter gradients against the original.
"""

import numpy as np
import torch
import torch.nn.functional as F


def _initial_layer(layer, H, e, e_dir):
    """Batched form of `initial_layer.forward` (which hard-codes batch size 1)."""
    uk = layer.fu(H).mean(dim=1)                              # (B, K, ch)
    e_h = F.normalize(e, p=1, dim=2)                          # (B, L, K)
    e_dir_n = F.normalize(e_dir, p=1, dim=2)                  # (B, 1, K)
    rl_ris = torch.matmul(e_h, uk)                            # (B, L, ch)
    rl_dir = torch.matmul(e_dir_n, uk)                        # (B, 1, ch)
    rl = layer.f(torch.cat((rl_ris, rl_dir.expand(-1, rl_ris.shape[1], -1)), dim=2))
    return uk, rl


def _max_excluding_self(uk):
    """Element-wise max over all other nodes, i.e. the A = 1 - I aggregation.

    Taking the top two values per feature lets every node reuse the global max
    unless that max is its own contribution, in which case the runner-up applies.
    """
    if uk.shape[1] == 1:
        return uk
    top2, idx = uk.topk(2, dim=1)                             # (B, 2, F)
    node = torch.arange(uk.shape[1], device=uk.device).view(1, -1, 1)
    is_self = idx[:, 0:1, :] == node                          # (B, K, F)
    return torch.where(is_self, top2[:, 1:2, :], top2[:, 0:1, :])


def _node_update_layer(layer, uk, rl, e, e_dir):
    """Batched form of `node_update_layer.forward`."""
    e_l = F.normalize(e.transpose(2, 1), p=1, dim=2)          # (B, K, L)
    mean_uk = torch.matmul(e_l, rl)                           # (B, K, F)
    max_user = _max_excluding_self(uk)
    uk_new = layer.fu(torch.cat((uk, max_user, mean_uk), dim=2))
    uk_update = torch.cat((uk_new, uk), dim=2)

    e_h = F.normalize(e, p=1, dim=2)
    e_dir_n = F.normalize(e_dir, p=1, dim=2)
    rl_ris = torch.matmul(e_h, uk)
    rl_dir = torch.matmul(e_dir_n, uk)
    mean_rl = torch.cat((rl_ris, rl_dir.expand(-1, rl_ris.shape[1], -1)), dim=2)
    rl_update = torch.cat((layer.f(torch.cat((rl, mean_rl), dim=2)), rl), dim=2)
    return uk_update, rl_update


def centralized_forward(model, user_feature, e, user_index, e_dir):
    """Batched equivalent of `node_update.forward(..., training=True)`.

    Inputs are not mutated, unlike the original, which zeroes them in place.
    """
    device = model.device
    mask = torch.as_tensor(user_index, dtype=torch.bool, device=device)
    m_f = mask.to(user_feature.dtype)
    B, K_tot = mask.shape
    n_ap = model.AP
    K_user = K_tot // n_ap

    uf = user_feature * m_f[:, None, :, None]
    e_m = e * m_f[:, None, :]
    e_dir_m = e_dir * m_f[:, None, :]

    uk, rl = _initial_layer(model.init_user, uf, e_m, e_dir_m)
    for layer in model.update_list:
        uk, rl = _node_update_layer(layer, uk, rl, e_m, e_dir_m)

    alpha = torch.cat([model.AP_coeff_NN_list[i](uk) for i in range(n_ap)], dim=1)

    W = model.BS_readout(uk) * m_f[:, None, :]                # (B, 2M, K_tot)
    twoM = W.shape[1]
    blocks = W.reshape(B, twoM, n_ap, K_user).permute(0, 2, 1, 3).reshape(B, n_ap, -1)
    blocks = F.normalize(blocks, dim=2, eps=1e-8) * torch.sqrt(model.Pmax * alpha).unsqueeze(2)
    W = blocks.reshape(B, n_ap, twoM, K_user).permute(0, 2, 1, 3).reshape(B, twoM, K_tot)

    ris_parts = []
    for bs in range(n_ap):
        e_ap = e_m[:, :, bs * K_user:(bs + 1) * K_user].reshape(B, -1)
        ris_parts.append(model.RIS_readout_AP_list[bs](rl, e_ap))
    theta = model.RIS_merge(ris_parts)
    return W, theta


def verify_equivalence(model, dataloader, K, threshold, device, atol=2e-4, check_grad=True):
    """Compare outputs (and optionally gradients) against the original forward."""
    import copy
    import numpy as np

    uf, e, ui, ed, _ = dataloader.gen_training_data(K, threshold, threshold, duplicate=False)
    uf, e, ed = uf.to(device), e.to(device), ed.to(device)
    ui_fast = np.array(ui, dtype=bool).copy()
    ref_model = copy.deepcopy(model)

    W_fast, th_fast = centralized_forward(model, uf.clone(), e.clone(), ui_fast, ed.clone())
    # The original mutates its inputs, so hand it private copies.
    W_ref, th_ref = ref_model(uf.clone(), e.clone(), np.array(ui, dtype=bool).copy(),
                              ed.clone(), training=True, duplicate=False)

    out = {
        "max_abs_diff_W": float((W_fast - W_ref).abs().max().item()),
        "max_abs_diff_theta": float((th_fast - th_ref).abs().max().item()),
        "rel_diff_W": float(((W_fast - W_ref).abs().max() / W_ref.abs().max()).item()),
        "outputs_match": None,
    }
    out["outputs_match"] = out["max_abs_diff_W"] < atol and out["max_abs_diff_theta"] < atol

    if check_grad:
        W_fast.sum().backward()
        W_ref.sum().backward()
        worst, worst_name = 0.0, ""
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), ref_model.named_parameters()):
            if p1.grad is None or p2.grad is None:
                continue
            d = (p1.grad - p2.grad).abs().max().item()
            if d > worst:
                worst, worst_name = d, n1
        out["max_abs_grad_diff"] = worst
        out["max_abs_grad_diff_param"] = worst_name
        model.zero_grad(set_to_none=True)
    return out


def decentralized_forward(model, user_feature, e, user_index, e_dir, include_cross_ap_csi=True):
    """Batched equivalent of `node_update.forward(..., training=False)`.

    Each AP still needs its own message-passing pass, so the AP loop stays, but
    the sample loop and the per-node loop are gone. The per-AP visibility mask of
    eq. (10) reduces to an elementwise AND: AP l sees link (l', k) exactly when it
    serves UE k and AP l' serves UE k too.
    """
    device = model.device
    n_ap = len(user_feature)
    B, n_ris, K_user = user_feature[0].shape[0], user_feature[0].shape[1], user_feature[0].shape[2]
    K_tot = K_user * n_ap

    ui = torch.stack([torch.as_tensor(np.asarray(u), dtype=torch.bool, device=device)
                      for u in user_index], dim=1)                     # (B, n_ap, K_user)
    uf_all = torch.cat(list(user_feature), dim=2)                      # (B, n_ris, K_tot, F)
    e_all = torch.cat(list(e), dim=2)                                  # (B, n_ris, K_tot)
    e_dir_all = torch.cat(list(e_dir), dim=2)                          # (B, 1, K_tot)

    ap_active = ui.any(dim=2)                                          # (B, n_ap)
    W = torch.zeros((B, 2 * model.M, K_tot), device=device)
    ris_parts = []

    for l in range(n_ap):
        if include_cross_ap_csi:
            vis = ui[:, l:l + 1, :] & ui                               # (B, n_ap, K_user)
        else:
            vis = torch.zeros_like(ui)
            vis[:, l, :] = ui[:, l, :]
        vis = vis.reshape(B, K_tot).to(uf_all.dtype)

        uk, rl = _initial_layer(model.init_user,
                                uf_all * vis[:, None, :, None],
                                e_all * vis[:, None, :],
                                e_dir_all * vis[:, None, :])
        e_m = e_all * vis[:, None, :]
        e_dir_m = e_dir_all * vis[:, None, :]
        for layer in model.update_list:
            uk, rl = _node_update_layer(layer, uk, rl, e_m, e_dir_m)

        alpha = model.AP_coeff_NN_list[l](uk)                          # (B, 1)
        served = torch.zeros((B, K_tot), device=device, dtype=uf_all.dtype)
        served[:, l * K_user:(l + 1) * K_user] = ui[:, l, :].to(uf_all.dtype)
        block = (model.BS_readout(uk) * served[:, None, :])[:, :, l * K_user:(l + 1) * K_user]
        block = block.reshape(B, -1)
        block = F.normalize(block, dim=1, eps=1e-8) * torch.sqrt(model.Pmax * alpha)
        W[:, :, l * K_user:(l + 1) * K_user] = block.reshape(B, 2 * model.M, K_user)

        # `e_AP` reads the unmasked per-AP edge weights, as in the original.
        e_ap = e[l].reshape(B, -1)
        part = model.RIS_readout_AP_list[l](rl, e_ap)                  # (B, n_ris, 4N)
        # An AP serving nobody is skipped upstream, so it must contribute nothing.
        ris_parts.append(part * ap_active[:, l].to(part.dtype)[:, None, None])

    theta = model.RIS_merge(ris_parts)
    return W, theta
