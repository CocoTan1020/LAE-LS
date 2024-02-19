"""Wrapper of AllenNLP model. Fixes errors based on model predictions"""
import logging
import os
import sys
from time import time

import torch
from allennlp.data.dataset import Batch
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.nn import util

from gector.bert_token_embedder import PretrainedBertEmbedder
# from gector.seq2labels_model import Seq2Labels
# from gector.seq2tag2seq_model import Seq2Tag2Seq
from gector.seq2tag_pipeline_model import Seq2Tag_pineline
from gector.tokenizer_indexer import PretrainedBertIndexer
from utils.helpers import PAD, UNK, get_target_sent_by_edits, START_TOKEN
from utils.helpers import get_weights_name

from transformers import BertTokenizer

logging.getLogger("werkzeug").setLevel(logging.ERROR)
logger = logging.getLogger(__file__)


class GecBERTModel(object):
    def __init__(self, vocab_path=None, model_paths=None,
                 weigths=None,
                 max_len=50,
                 min_len=3,
                 lowercase_tokens=False,
                 log=False,
                 iterations=1,
                 model_name='bert',
                 special_tokens_fix=1,
                 is_ensemble=True,
                 min_error_probability=0.0,
                 confidence=0,
                 del_confidence=0,
                 resolve_cycles=False,
                 pre_pretrained_model_path=None
                 ):
        self.model_weights = list(map(float, weigths)) if weigths else [1] * len(model_paths)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len
        self.min_len = min_len
        self.lowercase_tokens = lowercase_tokens
        self.min_error_probability = min_error_probability
        self.vocab = Vocabulary.from_files(vocab_path)
        self.log = log
        self.iterations = iterations
        self.confidence = confidence
        self.del_conf = del_confidence
        self.resolve_cycles = resolve_cycles
        self.pre_pretrained_model_path = pre_pretrained_model_path
        # set training parameters and operations

        self.indexers = []
        self.models = []
        self.tokenizer = BertTokenizer.from_pretrained(self.pre_pretrained_model_path)

        for model_path in model_paths:
            print('model_path', model_path)
            if is_ensemble:
                model_name, special_tokens_fix = self._get_model_data(model_path)
            weights_name = get_weights_name(model_name, lowercase_tokens)
            self.indexers.append(self._get_indexer(weights_name, special_tokens_fix))
            model = Seq2Tag_pineline(vocab=self.vocab,
                               text_field_embedder=self._get_embbeder(weights_name, special_tokens_fix),
                               confidence=self.confidence,
                               del_confidence=self.del_conf,
                               ).to(self.device)
            if torch.cuda.is_available():
                model.load_state_dict(torch.load(model_path), strict=True)
            else:
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')),
                                                 strict=True)
            model.eval()
            self.models.append(model)
            # print('OK')
            # exit()

    @staticmethod
    def _get_model_data(model_path):
        model_name = model_path.split('/')[-1]
        tr_model, stf = model_name.split('_')[:2]
        return tr_model, int(stf)

    def _convert(self, data):
        # print('data', data)
        all_class_probs = torch.zeros_like(data[0][0]['class_probabilities_labels'])
        # print('sum(self.model_weights)', sum(self.model_weights))  # 1
        for output, weight in zip(data, self.model_weights):
            # print('weight', weight)
            all_class_probs += weight * output[0]['class_probabilities_labels'] / sum(self.model_weights)
        # print('all_class_probs', all_class_probs)
        # print('error_probs', error_probs)
        # print('all_class_probs', all_class_probs)
        max_vals = torch.max(all_class_probs, dim=-1)
        # print('max_vals', max_vals)
        probs = max_vals[0].tolist()
        idx = max_vals[1].tolist()
        return probs, idx

    def predict(self, batches):
        for batch, model in zip(batches, self.models):
            batch = util.move_to_device(batch.as_tensor_dict(), 0 if torch.cuda.is_available() else -1)
            with torch.no_grad():
                prediction = model.forward(**batch, need_post_tag_batch=True, for_test=True)  # output_dict, orig_batch, post_tag_batch, tags
            break  # 不做模型集成
        logits_labels = prediction[0]['logits_labels']
        class_probabilities_labels = prediction[0]['class_probabilities_labels'].tolist()
        orig_batch = prediction[1]
        post_tag_batch = prediction[2]
        tags = prediction[3]
        return logits_labels, class_probabilities_labels, orig_batch, post_tag_batch, tags


    def _get_embbeder(self, weigths_name, special_tokens_fix):
        embedders = {'bert': PretrainedBertEmbedder(
            pretrained_model=weigths_name,
            requires_grad=False,
            top_layer_only=True,
            special_tokens_fix=special_tokens_fix,
            pre_pretrained_model_path=self.pre_pretrained_model_path
        )
        }
        text_field_embedder = BasicTextFieldEmbedder(
            token_embedders=embedders,
            embedder_to_indexer_map={"bert": ["bert", "bert-offsets"]},
            allow_unmatched_keys=True)
        return text_field_embedder

    def _get_indexer(self, weights_name, special_tokens_fix):
        bert_token_indexer = PretrainedBertIndexer(
            pretrained_model=weights_name,
            do_lowercase=self.lowercase_tokens,
            max_pieces_per_token=5,
            special_tokens_fix=special_tokens_fix,
            pre_pretrained_model_path=self.pre_pretrained_model_path
        )
        return {'bert': bert_token_indexer}

    def preprocess(self, token_batch):
        seq_lens = [len(sequence) for sequence in token_batch if sequence]
        if not seq_lens:
            return []
        max_len = min(max(seq_lens), self.max_len)
        batches = []
        for indexer in self.indexers:
            batch = []
            for sequence in token_batch:
                tokens = sequence[:max_len]
                # tokens = [Token(token) for token in ['$START'] + tokens]
                # tokens = [Token(token) for token in tokens]
                tokens = [Token(token) for token in ['[CLS]'] + tokens]
                batch.append(Instance({'tokens': TextField(tokens, indexer)}))
            batch = Batch(batch)
            batch.index_instances(self.vocab)
            batches.append(batch)

        return batches
    def postprocess_batch(self, batch, all_probabilities, all_idxs):
        # print('batch', batch)
        # print('all_probabilities', all_probabilities)
        # print('all_idxs', all_idxs)

        all_results = []
        all_tags = []
        for tokens, probabilities, idxs in zip(batch, all_probabilities, all_idxs):
            # print('tokens', tokens)
            # print('probabilities', probabilities)
            # print('idxs', idxs)
            tags = []
            results = []
            for i in range(len(idxs)):
                tag = self.vocab.get_token_from_index(idxs[i], namespace='pipeline_tags')
                tags.append(tag)
                if tag == '[DELETE]':
                    if self.vocab.get_token_from_index(int(tokens[i]), namespace='labels') == '[PAD]':
                        results.append('[PAD]')
                    else:
                        results.append('[MASK]')
                else:
                    results.append(self.vocab.get_token_from_index(int(tokens[i]), namespace='labels'))
            all_tags.append(tags)
            all_results.append(results)
            return all_tags, all_results

    def handle_batch(self, full_batch, need_probability=False):
        """
        Handle batch of requests.
        """
        final_batch = full_batch[:]
        batch_size = len(full_batch)
        prev_preds_dict = {i: [final_batch[i]] for i in range(len(final_batch))}
        short_ids = [i for i in range(len(full_batch))
                     if len(full_batch[i]) < self.min_len]
        pred_ids = [i for i in range(len(full_batch)) if i not in short_ids]
        total_updates = 0

        for n_iter in range(self.iterations):
            orig_batch = [final_batch[i] for i in pred_ids]
            # print('orig_batch', orig_batch)  # [['This', 'is', 'a', 'very', 'difficult', 'sentence', 'to', 'understand.']...]

            sequences = self.preprocess(orig_batch)
            # print('sequences', sequences)  # [<allennlp.data.dataset.Batch object at 0x0000018D33B1C908>]
            # for i in sequences:
            #     for j in i:
            #         print(j)

            # orig_batch = [i for i in sequences['tokens']]
            # print('orig_batch', orig_batch)

            if not sequences:
                print('error')
                break
            logits_labels, class_probabilities_labels, orig_batch, post_tag_batch, tags = self.predict(sequences)
        if need_probability:
            return orig_batch, post_tag_batch, tags, logits_labels, class_probabilities_labels
        else:
            return orig_batch, post_tag_batch, tags
