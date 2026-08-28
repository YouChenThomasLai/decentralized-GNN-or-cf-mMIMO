import numpy as np


DEFAULT_SQUARE_SIDE = 100 * np.sqrt(np.pi)
GATE_VERSION = "snapshot_network_scaling_v2_bpp_fixed_density"


def sample_square_bpp(count, side_length, rng):
    if count <= 0 or side_length <= 0:
        raise ValueError("count and side_length must be positive")
    return rng.uniform(0, side_length, size=(count, 2))


def wrapped_displacement(source, target, side_length):
    if side_length <= 0:
        raise ValueError("side_length must be positive")
    delta = np.asarray(target) - np.asarray(source)
    return (delta + side_length / 2) % side_length - side_length / 2


def wrapped_distance(source, target, side_length):
    return np.linalg.norm(
        wrapped_displacement(source, target, side_length), axis=-1
    )


def pairwise_wrapped_displacements(first, second, side_length):
    return wrapped_displacement(
        np.asarray(first)[..., None, :], np.asarray(second), side_length
    )


def pairwise_wrapped_distances(first, second, side_length):
    return np.linalg.norm(
        pairwise_wrapped_displacements(first, second, side_length), axis=-1
    )


def coordinates_in_square(locations, side_length):
    locations = np.asarray(locations)
    return bool(np.all((locations >= 0) & (locations < side_length)))


def association_diagnostics(association_mask):
    mask = np.asarray(association_mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError("association_mask must have shape (batch, UE, AP)")
    batch_size, num_users, num_ap = mask.shape
    serving_ap_count = mask.sum(axis=2)
    associated_ue_count = mask.sum(axis=1)
    global_links = mask.sum(axis=(1, 2))
    visible_links = (
        mask.transpose(0, 2, 1) * serving_ap_count[:, None, :]
    ).sum(axis=2)
    visibility_ratio = visible_links / np.maximum(global_links[:, None], 1)

    pair_shared = []
    for first in range(num_ap):
        for second in range(first + 1, num_ap):
            pair_shared.append(np.any(mask[:, :, first] & mask[:, :, second], axis=1))
    pair_shared = (
        np.stack(pair_shared, axis=1)
        if pair_shared
        else np.zeros((batch_size, 0), dtype=bool)
    )

    component_count = np.empty(batch_size, dtype=int)
    largest_component_fraction = np.empty(batch_size, dtype=float)
    for sample in range(batch_size):
        parent = np.arange(num_ap + num_users)

        def find(node):
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        def union(left, right):
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        for user, ap in np.argwhere(mask[sample]):
            union(ap, num_ap + user)
        roots = np.asarray([find(node) for node in range(len(parent))])
        _, sizes = np.unique(roots, return_counts=True)
        component_count[sample] = len(sizes)
        largest_component_fraction[sample] = sizes.max() / len(parent)

    active_ap_count = np.count_nonzero(associated_ue_count, axis=1)
    return {
        "serving_ap_count": serving_ap_count,
        "associated_ue_count": associated_ue_count,
        "total_associated_links": global_links,
        "visible_links_per_ap": visible_links,
        "local_to_global_visibility_ratio": visibility_ratio,
        "ap_pair_shared_ue": pair_shared,
        "connected_component_count": component_count,
        "largest_component_fraction": largest_component_fraction,
        "bipartite_fully_connected": component_count == 1,
        "active_ap_count": active_ap_count,
    }
