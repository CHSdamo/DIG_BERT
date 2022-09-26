import argparse
import random
import numpy as np
import torch

from datasets import load_dataset
from utils.scr import forward_func, nn_init, get_inputs, load_mappings
from utils.attributions import calculate_attributions, calculate_difference
from utils.paths import scale_inputs
from utils.metrics import eval_diff
from visualization import sum_of_cum_grads


def main(args):
    # set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model, tokenizer = nn_init(device, returns=True)

    auxiliary_data = load_mappings(knn_nbrs='500', distance='l2')

    # text = "uneasy mishmash of styles and genres ."
    text = "I like you. I love you."

    input_ids, baseline_ids, input_embed, baseline_embed, position_embed, baseline_position_embed, attention_mask \
        = get_inputs(text, device, args.baseline, auxiliary_data)

    scaled_features = scale_inputs(input_ids.squeeze().tolist(), baseline_ids.squeeze().tolist(), input_embed,
                                   baseline_embed, device, auxiliary_data, args.method, args.steps, args.strategy, args.factor)

    inputs = [scaled_features, input_ids, baseline_ids, input_embed, baseline_embed, position_embed,
              baseline_position_embed, attention_mask]

    cum_grads, norm_grads, norm_cum_grads = calculate_attributions(inputs, device, args, forward_func)

    ev = eval_diff(forward_func, input_embed, position_embed, norm_cum_grads[-1], attention_mask, device)
    print(f'The confidence of class {ev[1]} drops to {ev[0]*100}% after ablating the top{args.topk}% of token, using {args.method} and {args.baseline} baseline with {args.steps}')
    diff = calculate_difference(forward_func, cum_grads, input_embed, baseline_embed, attention_mask, position_embed)

    sum_of_cum_grads(torch.sum(cum_grads, dim=(1, 2)), diff, tokenizer, scaled_features, auxiliary_data, device, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IG Path')
    parser.add_argument('-steps', default=10, type=int)
    parser.add_argument('-topk', default=20, type=int)
    parser.add_argument('-factor', default=0, type=int)  # f
    parser.add_argument('-strategy', default='greedy', choices=['greedy', 'maxcount'],
                        help='The algorithm to find the next anchor point')
    parser.add_argument('-seed', default=42, type=int)
    parser.add_argument('-baseline', default='zero',
                        choices=['zero', 'constant', 'max', 'blurred', 'uniform'])
    parser.add_argument('-method', default='IG', choices=['IG', 'DIG'])

    args = parser.parse_args()

    main(args)
