import copy
import pickle
import torch
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DistilBertForMaskedLM, pipeline

model, tokenizer = None, None
word_idx_map, word_features, adj = [None] * 3


def nn_init(device, returns=False):
    global model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english",
                                                               return_dict=False)
    model.to(device)
    model.eval()
    model.zero_grad()

    if returns:
        return model, tokenizer


def forward_func(input_embed, attention_mask=None, position_embed=None, return_all_logits=False):
    global model
    embeds = model.distilbert.embeddings.dropout(model.distilbert.embeddings.LayerNorm(input_embed + position_embed))
    pred = predict(model, embeds, attention_mask=attention_mask)
    if return_all_logits:
        return pred
    else:
        return pred.max(1).values


def predict(model, inputs_embeds, attention_mask=None):
    return model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)[0]


def load_mappings(knn_nbrs='all', distance='l1'):
    from pathlib import Path
    pkl_path = Path(__file__).parent.parent.joinpath(f'./saves/distilbert_sst2_{knn_nbrs}_{distance}.pkl')
    with open(pkl_path, 'rb') as f:
        [word_idx_map, word_features, adj] = pickle.load(f)
    word_idx_map = dict(word_idx_map)
    return word_idx_map, word_features, adj


def get_pad_token_embed(device):
    global model
    return construct_word_embedding(model, torch.tensor([tokenizer.pad_token_id], device=device))


def construct_baseline_ids(text_ids, pad_token_id, sep_token_id, cls_token_id, baseline, text):
    global adj, word_features, tokenizer
    if adj is None:
        _, _, adj = load_mappings(knn_nbrs='500', distance='l2')
    if baseline == 'zero':
        baseline_ids = [cls_token_id] + [pad_token_id] * len(text_ids) + [sep_token_id]
    elif baseline == 'constant':
        baseline_ids = [cls_token_id] + [random.choice(text_ids)] * len(text_ids) + [sep_token_id]
    elif baseline == 'max':
        _, _, adj = load_mappings(knn_nbrs='500', distance='l1')
        baseline_ids = []
        for word_id in text_ids:
            baseline_id = adj[word_id].tocoo().col[-1]
            baseline_ids.append(baseline_id)
        baseline_ids = [cls_token_id] + baseline_ids + [sep_token_id]
    elif baseline == 'blurred':
        model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')
        input_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=False)
        baseline = []
        for i in range(len(input_ids[-1])):
            temp_input_ids = copy.deepcopy(input_ids)
            temp_input_ids[:, i] = tokenizer.mask_token_id
            token_logits = model(temp_input_ids)[0]
            mask_token_logits = token_logits[0, torch.tensor([i]), :]
            mask_token_logits = torch.softmax(mask_token_logits, dim=1)
            top_5 = torch.topk(mask_token_logits, 5, dim=1)
            if input_ids[:, i] == top_5.indices[0, 0]:
                baseline.append(top_5.indices[0, 1])
            else:
                baseline.append(top_5.indices[0, 0])
        baseline_ids = [tokenizer.cls_token_id] + baseline + [tokenizer.sep_token_id]
    elif baseline == 'uniform':
        baseline_ids = []
        for word_id in text_ids:
            baseline_id = random.choice(adj[word_id].tocoo().col)
            baseline_ids.append(baseline_id)
        baseline_ids = [cls_token_id] + baseline_ids + [sep_token_id]
    else:
        raise NotImplementedError
    return baseline_ids


def construct_input_bl_pair(tokenizer, text, pad_token_id, sep_token_id, cls_token_id, device, baseline):
    text_ids = tokenizer.encode(text, add_special_tokens=False, truncation=True,
                                max_length=tokenizer.max_len_single_sentence)
    input_ids = [cls_token_id] + text_ids + [sep_token_id]
    baseline_ids = construct_baseline_ids(text_ids, pad_token_id, sep_token_id, cls_token_id, baseline, text)

    return torch.tensor([input_ids], device=device), torch.tensor([baseline_ids], device=device)


def construct_input_bl_pos_id_pair(input_ids, device):
    global model
    seq_length = input_ids.size(1)
    position_ids = model.distilbert.embeddings.position_ids[:, 0:seq_length].to(device)
    baseline_position_ids = model.distilbert.embeddings.position_ids[:, 0:seq_length].to(device)

    return position_ids, baseline_position_ids


def construct_attention_mask(input_ids):
    return torch.ones_like(input_ids)


def construct_position_embedding(model, position_ids):
    return model.distilbert.embeddings.position_embeddings(position_ids)


def construct_word_embedding(model, input_ids):
    return model.distilbert.embeddings.word_embeddings(input_ids)


def construct_sub_embedding(model, input_ids, baseline_ids, position_ids, baseline_position_ids):
    input_embeddings = construct_word_embedding(model, input_ids)
    baseline_embeddings = construct_word_embedding(model, baseline_ids)
    input_position_embeddings = construct_position_embedding(model, position_ids)
    baseline_position_embeddings = construct_position_embedding(model, baseline_position_ids)

    return (input_embeddings, baseline_embeddings), \
           (input_position_embeddings, baseline_position_embeddings)


def get_inputs(text, device, baseline, auxiliary_data):
    global model, tokenizer, word_idx_map, word_features, adj
    pad_token_id = tokenizer.pad_token_id  # [PAD]
    sep_token_id = tokenizer.sep_token_id  # [SEP]
    cls_token_id = tokenizer.cls_token_id  # [CLS]

    word_idx_map, word_features, adj = auxiliary_data

    input_ids, baseline_ids = construct_input_bl_pair(tokenizer, text, pad_token_id, sep_token_id, cls_token_id,
                                                      device, baseline)
    position_ids, baseline_position_ids = construct_input_bl_pos_id_pair(input_ids, device)
    attention_mask = construct_attention_mask(input_ids)

    (input_embed, baseline_embed), (position_embed, baseline_position_embed) = construct_sub_embedding(model, input_ids,
                                                                                                       baseline_ids,
                                                                                                       position_ids,
                                                                                                       baseline_position_ids)

    return [input_ids, baseline_ids, input_embed, baseline_embed, position_embed, baseline_position_embed,
            attention_mask]
