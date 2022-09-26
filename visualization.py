import matplotlib.pyplot as plt
import torch


def sum_of_cum_grads(values, diff, tokenizer, scaled_features, auxiliary_data, device, args):
    # round(values.max().item())
    plt.plot(values.cpu().detach().numpy())
    plt.axhline(y=diff.cpu().detach().numpy(), color='r', linestyle='-')
    _, sentences = closest_token(tokenizer, scaled_features, auxiliary_data, device)
    plt.text(-1.0, -0.1, sentences[0], fontsize=6)
    plt.title('Sum of Cumulative Grads')
    plt.text(round((args.steps+2) / 5), round(values.max().item()), sentences[round((args.steps+2) / 5)], fontsize=6)
    plt.text(round((args.steps+2) / 1.5), round(diff.item()), sentences[-1], fontsize=6)
    plt.xlabel('Number of Samples')
    plt.ylabel('Mean Cumulative Sum of Gradients')
    plt.show()


def closest_token(tokenizer, scaled_features, auxiliary_data, device):
    word_embeddings = torch.from_numpy(auxiliary_data[1]).to(device)
    interpol_tokens = []
    sentences = []
    for embed in scaled_features:
        diff = embed.unsqueeze(1).expand([embed.shape[0], word_embeddings.shape[0], embed.shape[-1]]) - \
                  word_embeddings.expand([embed.shape[0], word_embeddings.shape[0], word_embeddings.shape[1]])
        token_id = torch.argmin(torch.sqrt(torch.sum(diff ** 2, dim=-1)), dim=1)
        token = tokenizer.convert_ids_to_tokens(token_id)
        sentence = tokenizer.convert_tokens_to_string(token)
        interpol_tokens.append(token)
        sentences.append(sentence)
    return interpol_tokens, sentences


