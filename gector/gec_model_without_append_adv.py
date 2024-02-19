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
from gector.seq2tag2seq_model import Seq2Tag2Seq
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
                 iterations=3,
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
            if is_ensemble:
                model_name, special_tokens_fix = self._get_model_data(model_path)
            weights_name = get_weights_name(model_name, lowercase_tokens)
            self.indexers.append(self._get_indexer(weights_name, special_tokens_fix))
            model = Seq2Tag2Seq(vocab=self.vocab,
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

    @staticmethod
    def _get_model_data(model_path):
        model_name = model_path.split('/')[-1]
        tr_model, stf = model_name.split('_')[:2]
        return tr_model, int(stf)

    def _restore_model(self, input_path):
        if os.path.isdir(input_path):
            print("Model could not be restored from directory", file=sys.stderr)
            filenames = []
        else:
            filenames = [input_path]
        for model_path in filenames:
            try:
                if torch.cuda.is_available():
                    loaded_model = torch.load(model_path)
                else:
                    loaded_model = torch.load(model_path,
                                              map_location=lambda storage,
                                                                  loc: storage)
            except:
                print(f"{model_path} is not valid model", file=sys.stderr)
            own_state = self.model.state_dict()
            for name, weights in loaded_model.items():
                if name not in own_state:
                    continue
                try:
                    if len(filenames) == 1:
                        own_state[name].copy_(weights)
                    else:
                        own_state[name] += weights
                except RuntimeError:
                    continue
        print("Model is restored", file=sys.stderr)

    def predict(self, batches):
        t11 = time()
        predictions = []
        for batch, model in zip(batches, self.models):
            batch = util.move_to_device(batch.as_tensor_dict(), 0 if torch.cuda.is_available() else -1)
            with torch.no_grad():
                prediction = model.forward(**batch)
            predictions.append(prediction)
        # print('predictions', predictions)  # [{logits_labels, logits_d_tags, class_probabilities_labels, class_probabilities_d_tags, max_error_probability}]
        preds, idx, error_probs = self._convert(predictions)
        # print('preds', preds)  # [[0.1935310959815979, 0.19127970933914185, 0.13783395290374756, 0.15542078018188477, 0.02418181300163269, 0.33387303352355957, 0.4226502478122711, 0.2857482433319092, 0.07311517000198364], [0.2830435037612915, 0.29013288021087646, 0.27518415451049805, 0.28838878870010376, 0.01005860697478056, 0.13507406413555145, 0.18534481525421143, 0.3115081191062927, 0.208210289478302], [0.23968571424484253, 0.2520170211791992, 0.21381646394729614, 0.172230064868927, 0.07303279638290405, 0.1138685941696167, 0.7328836917877197, 0.22187566757202148, 0.08232158422470093]]
        # print('idx', idx)  # idx [[0, 0, 0, 0, 395, 395, 6, 0, 5000], [0, 0, 0, 0, 5000, 5000, 6, 0, 0], [0, 0, 0, 0, 0, 0, 6, 0, 0]]
        # print('error_probs', error_probs)  # error_probs [0.6651961207389832, 0.8221973180770874, 0.587377667427063]
        t55 = time()
        if self.log:
            print(f"Inference time {t55 - t11}")
        return preds, idx, error_probs

    def get_token_action(self, token, index, prob, sugg_token):
        """Get lost of suggested actions for token."""
        # cases when we don't need to do anything
        if sugg_token in [UNK, PAD, '$KEEP']:
            return None
        # if sugg_token.startswith('$REPLACE_') or sugg_token.startswith('$TRANSFORM_') or sugg_token == '$DELETE':
        #     start_pos = index
        #     end_pos = index + 1
        # elif sugg_token.startswith("$APPEND_") or sugg_token.startswith("$MERGE_"):
        #     start_pos = index + 1
        #     end_pos = index + 1
        # if sugg_token == "$DELETE":
        #     sugg_token_clear = ""
        # elif sugg_token.startswith('$TRANSFORM_') or sugg_token.startswith("$MERGE_"):
        #     sugg_token_clear = sugg_token[:]
        # else:
        #     sugg_token_clear = sugg_token[sugg_token.index('_') + 1:]

        if sugg_token == "$DELETE":
            sugg_token_clear = ""
            start_pos = index
            end_pos = index + 1
        else:
            start_pos = index
            end_pos = index + 1
            sugg_token_clear = sugg_token[:]

        return start_pos, end_pos, sugg_token_clear, prob

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

    def _convert(self, data):
        # print('data', data)
        all_class_probs = torch.zeros_like(data[0][0]['class_probabilities_labels'])
        error_probs = torch.zeros_like(data[0][0]['max_error_probability'])
        # print('sum(self.model_weights)', sum(self.model_weights))  # 1
        for output, weight in zip(data, self.model_weights):
            # print('weight', weight)
            all_class_probs += weight * output[0]['class_probabilities_labels'] / sum(self.model_weights)
            error_probs += weight * output[0]['max_error_probability'] / sum(self.model_weights)
        # print('all_class_probs', all_class_probs)
        # print('error_probs', error_probs)
        max_vals = torch.max(all_class_probs, dim=-1)
        # print('max_vals', max_vals)
        '''
        max_vals torch.return_types.max(
        values=tensor([[0.1935, 0.1913, 0.1378, 0.1554, 0.0242, 0.3339, 0.4227, 0.2857, 0.0731],
        [0.2830, 0.2901, 0.2752, 0.2884, 0.0101, 0.1351, 0.1853, 0.3115, 0.2082],
        [0.2397, 0.2520, 0.2138, 0.1722, 0.0730, 0.1139, 0.7329, 0.2219, 0.0823]]),
        indices=tensor([[   0,    0,    0,    0,  395,  395,    6,    0, 5000],
        [   0,    0,    0,    0, 5000, 5000,    6,    0,    0],
        [   0,    0,    0,    0,    0,    0,    6,    0,    0]]))
        '''
        probs = max_vals[0].tolist()
        idx = max_vals[1].tolist()
        return probs, idx, error_probs.tolist()

    def update_final_batch(self, final_batch, pred_ids, pred_batch,
                           prev_preds_dict):
        new_pred_ids = []
        total_updated = 0
        for i, orig_id in enumerate(pred_ids):
            orig = final_batch[orig_id]
            pred = pred_batch[i]
            prev_preds = prev_preds_dict[orig_id]
            if orig != pred and pred not in prev_preds:
                final_batch[orig_id] = pred
                new_pred_ids.append(orig_id)
                prev_preds_dict[orig_id].append(pred)
                total_updated += 1
            elif orig != pred and pred in prev_preds:
                # update final batch, but stop iterations
                final_batch[orig_id] = pred
                total_updated += 1
            else:
                continue
        return final_batch, new_pred_ids, total_updated

    def postprocess_batch(self, batch, all_probabilities, all_idxs, error_probs):
        all_results = []
        noop_index = self.vocab.get_token_index("$KEEP", "labels")
        # print('noop_index', noop_index)
        for tokens, probabilities, idxs, error_prob in zip(batch,
                                                           all_probabilities,
                                                           all_idxs,
                                                           error_probs):
            # print('tokens', tokens)
            # print('probabilities', probabilities)
            # print('idxs', idxs)

            # test:
            # tokens = ['This', 'is', 'a', 'very', 'difficult', 'sentence', 'to', 'understand.']
            # idxs = [1, 2, 2, 1, 1, 1, 1, 1, 1]
            # tokens = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h.']
            # idxs = [0, 0, 1, 2, 3, 3, 4, 5, 5000]
            # print('tokens', tokens)
            # # print('probabilities', probabilities)
            # print('idxs', idxs)

            length = min(len(tokens), self.max_len)
            edits = []

            # skip whole sentences if there no errors
            if max(idxs) == 0:
                all_results.append(tokens)
                continue

            # # skip whole sentence if probability of correctness is not high
            # if error_prob < self.min_error_probability:
            #     all_results.append(tokens)
            #     continue

            for i in range(length):
                token = tokens[i]
                # skip if there is no error
                if idxs[i] == noop_index:
                    continue

                sugg_token = self.vocab.get_token_from_index(idxs[i],
                                                             namespace='labels')
                # print('sugg_token', sugg_token)
                action = self.get_token_action(token, i, probabilities[i],
                                               sugg_token)
                # print('action', action)
                if not action:
                    continue

                edits.append(action)
            all_results.append(get_target_sent_by_edits(tokens, edits))
            # print('after', get_target_sent_by_edits(tokens, edits))
        return all_results

    def handle_batch(self, full_batch):
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
            # for seq in sequences:
                # for s in seq:
                    # print('s', s)

            if not sequences:
                print('error')
                break
            probabilities, idxs, error_probs = self.predict(sequences)
            # print('probabilities', probabilities)  # [[0.1935310959815979, 0.19127970933914185, 0.13783395290374756, 0.15542078018188477, 0.02418181300163269, 0.33387303352355957, 0.4226502478122711, 0.2857482433319092, 0.07311517000198364], [0.2830435037612915, 0.29013288021087646, 0.27518415451049805, 0.28838878870010376, 0.01005860697478056, 0.13507406413555145, 0.18534481525421143, 0.3115081191062927, 0.208210289478302], [0.23968571424484253, 0.2520170211791992, 0.21381646394729614, 0.172230064868927, 0.07303279638290405, 0.1138685941696167, 0.7328836917877197, 0.22187566757202148, 0.08232158422470093]]
            # print('idxs', idxs)  # idxs [[0, 0, 0, 0, 395, 395, 6, 0, 5000], [0, 0, 0, 0, 5000, 5000, 6, 0, 0], [0, 0, 0, 0, 0, 0, 6, 0, 0]]
            # print('error_probs', error_probs)  # error_probs [0.6651961207389832, 0.8221973180770874, 0.587377667427063]

            final_batch = []
            for idx in idxs:
                token = []
                for id in idx[1:]:
                    t = self.tokenizer._convert_id_to_token(id)
                    if t != '[PAD]':
                        token.append(t)
                # print('token', token)
                sentence = self.tokenizer.convert_tokens_to_string(token)
                # print('sentence', sentence)
                final_batch.append(sentence)

            # pred_batch = self.postprocess_batch(orig_batch, probabilities,
            #                                     idxs, error_probs)
            # final_batch, pred_ids, cnt = \
            #     self.update_final_batch(final_batch, pred_ids, pred_batch,
            #                             prev_preds_dict)
            # print('final_batch', final_batch)
            # print('pred_ids', pred_ids)
            # print('cnt', cnt)

            # total_updates += cnt

            # if not pred_ids:
            #     break
        total_updates = 0
        return final_batch, total_updates
