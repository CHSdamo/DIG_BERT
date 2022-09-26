import argparse
import pickle
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.neighbors import kneighbors_graph


def main(args):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english",
                                                               return_dict=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    model.zero_grad()

    def get_word_embeddings():
        global model
        return model.distilbert.embeddings.word_embeddings.weight

    print(f'Starting KNN computation..')

    word_features = get_word_embeddings().cpu().detach().numpy()
    word_idx_map = tokenizer.get_vocab()

    if args.distance == 'l1':
        A = kneighbors_graph(word_features, args.nbrs, mode='distance', p=1, n_jobs=-1)  # l1 distance
    elif args.distance == 'l2':
        A = kneighbors_graph(word_features, args.nbrs, mode='distance', n_jobs=-1)  # l2 distance

    knn_fname = f'./saves/distilbert_sst2_{args.nbrs}_{args.distance}.pkl'
    with open(knn_fname, 'wb') as f:
        pickle.dump([word_idx_map, word_features, A], f)

    print(f'Written KNN data at {knn_fname}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='knn')
    parser.add_argument('-nbrs', default=500, type=int)
    parser.add_argument('-distance', default='l2', choices=['l1', 'l2'])

    args = parser.parse_args()

    main(args)
