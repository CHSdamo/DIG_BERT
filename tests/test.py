import unittest
import torch
from utils.attributions import custom_integrated_gradients
from utils.scr import nn_init, forward_func, load_mappings, construct_input_bl_pair, construct_word_embedding
from utils.paths import scale_inputs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model, tokenizer = nn_init(device, returns=True)

pad_token_id = tokenizer.pad_token_id
sep_token_id = tokenizer.sep_token_id
cls_token_id = tokenizer.cls_token_id

auxiliary_data = load_mappings(knn_nbrs='500', distance='l2')

input_ids = [101, 1045, 2066, 2017, 1012, 1045, 2293, 2017, 1012, 102]
baseline_ids = [101, 0, 0, 0, 0, 0, 0, 0, 0, 102]
attention_mask = torch.ones_like(torch.tensor([input_ids], device=device))
input_embed = construct_word_embedding(model, torch.tensor([input_ids], device=device))
baseline_embed = construct_word_embedding(model, torch.tensor([baseline_ids], device=device))
position_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], device='cuda:0')
position_embed = model.distilbert.embeddings.position_embeddings(position_ids)


class MyTestCase(unittest.TestCase):
    def test_zero_baseline_shape(self):
        baseline = 'zero'
        input_ids, baseline_ids = construct_input_bl_pair(tokenizer, text, pad_token_id, sep_token_id, cls_token_id,
                                                          device, baseline)
        self.assertTrue(isinstance(input_ids, torch.Tensor))
        self.assertTrue(isinstance(baseline_ids, torch.Tensor))
        self.assertEqual(input_ids.shape, (1, 11))
        self.assertEqual(baseline_ids.shape, (1, 11))

    def test_constant_baseline_shape(self):
        baseline = 'constant'
        input_ids, baseline_ids = construct_input_bl_pair(tokenizer, text, pad_token_id, sep_token_id, cls_token_id,
                                                          device, baseline)
        self.assertTrue(isinstance(input_ids, torch.Tensor))
        self.assertTrue(isinstance(baseline_ids, torch.Tensor))
        self.assertEqual(input_ids.shape, (1, 11))
        self.assertEqual(baseline_ids.shape, (1, 11))

    def test_max_baseline_shape(self):
        baseline = 'max'
        input_ids, baseline_ids = construct_input_bl_pair(tokenizer, text, pad_token_id, sep_token_id, cls_token_id,
                                                          device, baseline)
        self.assertTrue(isinstance(input_ids, torch.Tensor))
        self.assertTrue(isinstance(baseline_ids, torch.Tensor))
        self.assertEqual(input_ids.shape, (1, 11))
        self.assertEqual(baseline_ids.shape, (1, 11))

    def test_blurred_baseline_shape(self):
        baseline = 'blurred'
        input_ids, baseline_ids = construct_input_bl_pair(tokenizer, text, pad_token_id, sep_token_id, cls_token_id,
                                                          device, baseline)
        self.assertTrue(isinstance(input_ids, torch.Tensor))
        self.assertTrue(isinstance(baseline_ids, torch.Tensor))
        self.assertEqual(input_ids.shape, (1, 11))
        self.assertEqual(baseline_ids.shape, (1, 11))

    def test_uniform_baseline_shape(self):
        baseline = 'uniform'
        input_ids, baseline_ids = construct_input_bl_pair(tokenizer, text, pad_token_id, sep_token_id, cls_token_id,
                                                          device, baseline)
        self.assertTrue(isinstance(input_ids, torch.Tensor))
        self.assertTrue(isinstance(baseline_ids, torch.Tensor))
        self.assertEqual(input_ids.shape, (1, 11))
        self.assertEqual(baseline_ids.shape, (1, 11))

    def test_dig_shape(self):
        global auxiliary_data
        steps = 20
        method = 'DIG'
        strategy = 'greedy'
        scaled_features = scale_inputs(input_ids, baseline_ids, input_embed, baseline_embed, device, auxiliary_data,
                                       method=method,steps=steps, strategy=strategy)
        self.assertEqual(scaled_features.shape, (steps + 2, len(input_ids), 768))

    def test_ig_shape(self):
        global auxiliary_data
        steps = 20
        method = 'IG'
        strategy = 'greedy'
        scaled_features = scale_inputs(input_ids, baseline_ids, input_embed, baseline_embed, device, auxiliary_data,
                                       method=method, steps=steps, strategy=strategy)
        self.assertEqual(scaled_features.shape, (steps + 2, len(input_ids), 768))

        cum_grads, norm_grads, norm_cum_grads = custom_integrated_gradients(forward_func, scaled_features, input_embed,
                                                                            baseline_embed, position_embed,
                                                                            attention_mask)
        self.assertEqual(cum_grads.shape, (steps + 2, len(input_ids), 768))
        self.assertEqual(norm_grads.shape, (steps + 2, len(input_ids)))
        self.assertEqual(norm_cum_grads.shape, (steps + 2, len(input_ids)))


if __name__ == '__main__':
    unittest.main()
