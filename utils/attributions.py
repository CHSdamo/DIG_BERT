import torch
from utils.dig import DiscretetizedIntegratedGradients


def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions


def run_dig_explanation(dig_func, all_input_embed, position_embed, attention_mask, steps):
    attributions, scaled_grads = dig_func.attribute(scaled_features=all_input_embed,
                                                    additional_forward_args=(attention_mask, position_embed),
                                                    n_steps=steps)
    scaled_grads_word = summarize_attributions(scaled_grads)

    cum_grad = scaled_grads[0].unsqueeze(0)
    norm_cum_grad = summarize_attributions(cum_grad).unsqueeze(0)
    norm_cum_grads = torch.clone(norm_cum_grad)
    cum_grads = torch.clone(cum_grad)
    for i in range(len(scaled_grads) - 1):
        cum_grad += scaled_grads[i + 1].unsqueeze(0)
        norm_cum_grad = summarize_attributions(cum_grad).unsqueeze(0)
        norm_cum_grads = torch.cat((norm_cum_grads, norm_cum_grad))
        cum_grads = torch.cat((cum_grads, cum_grad))

    return cum_grads, scaled_grads_word, norm_cum_grads


def run_ig_explanation(ig_func, all_input_embed, position_embed, attention_mask, steps):
    attributions = ig_func.attribute(scaled_features=all_input_embed,
                                     additional_forward_args=(attention_mask, position_embed), n_steps=steps)
    attributions_word = summarize_attributions(attributions)

    return attributions_word


def custom_integrated_gradients(forward_func, scaled_features, input_embed, baseline_embed, position_embed,
                                attention_mask):
    pred = torch.argmax(forward_func(input_embed, attention_mask=attention_mask, position_embed=position_embed,
                                     return_all_logits=True)).item()
    grads = []              # grad for each interpol
    norm_grads = []         # every Interpolation norm grad for visul   (22, 11)
    cum_grads = []          # for Sum of Cumulative Grads at each interpol (22, 11, 768)
    norm_cum_grads = []     # norm Cumulative Grads for visual at each interpol  (22, 11)
    for i in range(len(scaled_features)):
        scaled_feature = scaled_features[i].detach()
        scaled_feature.requires_grad = True
        output = forward_func(scaled_feature, attention_mask=attention_mask, position_embed=position_embed,
                              return_all_logits=True)
        grad = torch.autograd.grad(output[:, pred], scaled_feature)[0]
        grads.append(grad)
        cum_grad = (input_embed - baseline_embed) * (torch.stack(grads)).mean(0)
        cum_grads.append(cum_grad)
        norm_grad = summarize_attributions(grad)
        norm_grads.append(norm_grad)
        norm_cum_grad = summarize_attributions(cum_grad)
        norm_cum_grads.append(norm_cum_grad)

    return torch.stack(cum_grads).squeeze(1), torch.stack(norm_grads), torch.stack(norm_cum_grads)


def calculate_difference(forward_func, cum_grads, input_embed, baseline_embed, attention_mask, position_embed):
    pred = torch.argmax(forward_func(input_embed, attention_mask=attention_mask, position_embed=position_embed,
                                     return_all_logits=True)).item()
    p1 = forward_func(input_embed, attention_mask=attention_mask, position_embed=position_embed,
                      return_all_logits=True)
    p2 = forward_func(baseline_embed, attention_mask=attention_mask, position_embed=position_embed,
                      return_all_logits=True)
    diff = p1.squeeze()[pred] - p2.squeeze()[pred]
    satu = torch.sum(cum_grads[-1])
    return diff


# computes the attributions for given input
def calculate_attributions(inputs, device, args, forward_func):
    # move inputs to main device
    scaled_features, input_ids, baseline_ids, input_embed, baseline_embed, position_embed, \
     baseline_position_embed, attention_mask = [x.to(device) if x is not None else None for x in inputs]

    # compute attribution
    if args.method == 'IG':
        cum_grads, norm_grads, norm_cum_grads = custom_integrated_gradients(forward_func, scaled_features,
                                                                            input_embed, baseline_embed,
                                                                            position_embed,
                                                                            attention_mask)

    elif args.method == 'DIG':
        attr_func = DiscretetizedIntegratedGradients(forward_func)
        cum_grads, norm_grads, norm_cum_grads = run_dig_explanation(attr_func, scaled_features, position_embed,
                                                                    attention_mask,
                                                                    (2 ** args.factor) * (args.steps + 1) + 1)

    else:
        raise NotImplementedError

    return cum_grads, norm_grads, norm_cum_grads
