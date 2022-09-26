import sys
import torch
import numpy as np
from copy import deepcopy
from collections import defaultdict as ddict

sys.path.append('../')

word_idx_map, word_features, adj = [None] * 3


def distance(A, B):
    return np.sqrt(np.sum((A - B) ** 2))


def find_next_word(word_idx, bl_idx, word_path, strategy='greedy', steps=30):
    global adj, word_features

    if word_idx == bl_idx:
        return bl_idx

    anchor_map = ddict(list)
    cx = adj[word_idx].tocoo()

    for j, v in zip(cx.col, cx.data):
        # we should not consider the anchor word to be the ref_idx [baseline] unless forced to.
        if j == bl_idx:
            continue

        if strategy == 'greedy':
            # calculate the distance of the monotonized vec from the anchor point
            # word_features[j] 离 wrd_idx 最近的点的embd 先做为 anchor
            monotonic_vec = make_monotonic_vec(word_features[bl_idx], word_features[word_idx], word_features[j], steps)
            anchor_map[j] = [distance(word_features[j], monotonic_vec)]  # l2 distance
        elif strategy == 'maxcount':
            non_mono_count = 10000 - monotonic(word_features[bl_idx], word_features[word_idx], word_features[j],
                                               ret='count')
            anchor_map[j] = [non_mono_count]
        else:
            raise NotImplementedError

    if len(anchor_map) == 0:
        return bl_idx
    # distance from small to big
    sorted_dist_map = {k: v for k, v in sorted(anchor_map.items(), key=lambda item: item[1][0])}

    # remove words that are already selected in the path
    for key in word_path:
        sorted_dist_map.pop(key, None)

    if len(sorted_dist_map) == 0:
        return bl_idx

    # return the top key
    return next(iter(sorted_dist_map))


def find_word_path(word_idx, bl_idx, steps=30, strategy='greedy'):
    global word_idx_map

    # if wrd_idx is CLS or SEP then just copy that and return
    if ('[CLS]' in word_idx_map and word_idx == word_idx_map['[CLS]']) or (
            '[SEP]' in word_idx_map and word_idx == word_idx_map['[SEP]']):
        return [word_idx] * (steps + 1)

    word_path = [word_idx]
    last_idx = word_idx
    for step in range(steps):
        next_idx = find_next_word(last_idx, bl_idx, word_path, strategy=strategy, steps=steps)
        word_path.append(next_idx)
        last_idx = next_idx
    return word_path


def get_alphas(steps, device):
    return torch.linspace(0, 1, steps + 2, device=device)


def get_path(input, baseline, steps, device):
    alphas = get_alphas(steps, device)
    paths = baseline + alphas.reshape(-1, 1, 1) * (input - baseline)

    return paths


def monotonic(vec1, vec2, vec3, ret='bool'):
    # check if vec3 [inpterpolation] is monotonic w.r.t. vec1 [baseline] and vec2 [input]
    # i.e., vec3 should lie between vec1 and vec2 (for both +ve and -ve cases)

    increasing_dims = vec1 > vec2  # dims where baseline > input
    decreasing_dims = vec1 < vec2  # dims where baseline < input
    equal_dims = vec1 == vec2  # dims where baseline == input

    vec3_greater_vec1 = vec3 >= vec1
    vec3_greater_vec2 = vec3 >= vec2
    vec3_lesser_vec1 = vec3 <= vec1
    vec3_lesser_vec2 = vec3 <= vec2
    vec3_equal_vec1 = vec3 == vec1
    vec3_equal_vec2 = vec3 == vec2

    # if, for some dim: vec1 > vec2 then vec1 >= vec3 >= vec2
    # elif: vec1 < vec2 then vec1 <= vec3 <= vec2
    # elif: vec1 == vec2 then vec1 == vec3 == vec2
    valid = (increasing_dims * vec3_lesser_vec1 * vec3_greater_vec2
             + decreasing_dims * vec3_greater_vec1 * vec3_lesser_vec2
             + equal_dims * vec3_equal_vec1 * vec3_equal_vec2)

    if ret == 'bool':
        return valid.sum() == vec1.shape[0]
    elif ret == 'count':
        return valid.sum()
    elif ret == 'vec':
        return valid


def make_monotonic_vec(vec1, vec2, vec3, steps):
    # create a new vec4 from vec3 [anchor] which is monotonic w.r.t. vec1 [baseline] and vec2 [input]

    mono_dims = monotonic(vec1, vec2, vec3, ret='vec')
    non_mono_dims = ~mono_dims

    if non_mono_dims.sum() == 0:
        return vec3

    # make vec4 monotonic
    vec4 = deepcopy(vec3)
    vec4[non_mono_dims] = vec2[non_mono_dims] - (1.0 / steps) * (vec2[non_mono_dims] - vec1[non_mono_dims])

    return vec4


def make_monotonic_path(word_path_ids, ref_idx, steps=30, factor=0):
    global word_features
    monotonic_embeds = [word_features[word_path_ids[0]]]
    vec1 = word_features[ref_idx]  # baseline embed

    for idx in range(len(word_path_ids) - 1):
        vec2 = monotonic_embeds[-1]                   # last word in path
        vec3 = word_features[word_path_ids[idx + 1]]  # next word in path
        vec4 = make_monotonic_vec(vec1, vec2, vec3, steps)
        monotonic_embeds.append(vec4)
    monotonic_embeds.append(vec1)

    # reverse the list so that baseline is the first and input word is the last
    monotonic_embeds.reverse()

    final_embeds = monotonic_embeds

    # do upscaling for factor number of times
    for _ in range(factor):
        final_embeds = upscale(final_embeds)

    # verify monotonicity
    check = True
    for i in range(len(final_embeds) - 1):
        check *= monotonic(final_embeds[-1], final_embeds[i], final_embeds[i + 1], ret='bool')
    assert check

    return final_embeds


def upscale(embs):
    # add a average embedding between each consecutive vec in embs
    embs = np.array(embs)
    avg_embs = 0.5 * (embs[0:-1] + embs[1:])
    final_embs = np.empty((embs.shape[0] + avg_embs.shape[0], embs.shape[1]), dtype=embs.dtype)
    final_embs[::2] = embs
    final_embs[1::2] = avg_embs

    return final_embs


def scale_inputs(input_ids, baseline_ids, input_embed, baseline_embed, device, auxiliary_data, method='IG', steps=30,
                 strategy='greedy', factor=0):
    global word_idx_map, word_features, adj
    word_idx_map, word_features, adj = auxiliary_data
    all_path_embeds = []

    # generates the paths required by IG
    if method == 'IG':
        all_path_embeds = get_path(input_embed, baseline_embed, steps, device)

    # generates the paths required by DIG
    elif method == 'DIG':
        for idx in range(len(input_ids)):
            word_path = find_word_path(input_ids[idx], baseline_ids[idx], steps=steps, strategy=strategy)
            monotonic_embeds = make_monotonic_path(word_path, baseline_ids[idx], steps=steps, factor=factor)
            all_path_embeds.append(monotonic_embeds)
        all_path_embeds = torch.tensor(np.stack(all_path_embeds, axis=1), dtype=torch.float, device=device,
                                       requires_grad=True)
    else:
        raise NotImplementedError

    return all_path_embeds
