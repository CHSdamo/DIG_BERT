import torch
from utils.scr import get_pad_token_embed


def eval_diff(forward_fn, input_embed, position_embed, attr, attention_mask, device):
    baseline_token_embed = get_pad_token_embed(device)
    logits_original = forward_fn(input_embed, attention_mask=attention_mask, position_embed=position_embed,
                                 return_all_logits=True).squeeze()
    pred_label = torch.argmax(logits_original).item()
    prob_original = torch.softmax(logits_original, dim=0)

    local_input_embed = topk_ablation(input_embed, baseline_token_embed, attr, topk=20)

    logits_perturbed = forward_fn(local_input_embed, attention_mask=attention_mask, position_embed=position_embed,
                                  return_all_logits=True).squeeze()
    prob_perturbed = torch.softmax(logits_perturbed, dim=0)

    # (torch.log(prob_perturbed[pred_label]) - torch.log(prob_original[pred_label])).item(), pred_label
    return logits_perturbed[pred_label] / logits_original[pred_label], pred_label


def get_topk_indices(attr, topk=20):
    topk_indices = torch.topk(attr, int(attr.shape[0] * topk / 100), sorted=False).indices
    return topk_indices


def topk_ablation(input_embed, base_token_emb, attr, topk=20):
    topk_indices = get_topk_indices(attr, topk)
    local_input_embed = input_embed.detach().clone()
    local_input_embed[0][topk_indices] = base_token_emb
    return local_input_embed
