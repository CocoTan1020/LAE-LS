"""
Generator model.
Predict tags for every token based on the original texts.
Transfer the original texts into the terminal texts based on original tokens and tags.
[original sequences -> tags -> terminal sequences]
"""

from typing import Dict, Optional, List, Any

import numpy
import torch
import torch.nn.functional as F
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TimeDistributed, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy
from overrides import overrides
from torch.nn.modules.linear import Linear
from utils.helpers import PAD, UNK, get_target_sent_by_edits, START_TOKEN


@Model.register("seq2tag2seq")
class Seq2Tag2Seq(Model):
    """
    This ``Seq2Labels`` simply encodes a sequence of text with a stacked ``Seq2SeqEncoder``, then
    predicts a tag (or couple tags) for each token in the sequence.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    encoder : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use in between embedding tokens
        and predicting output tags.
    calculate_span_f1 : ``bool``, optional (default=``None``)
        Calculate span-level F1 metrics during training. If this is ``True``, then
        ``label_encoding`` is required. If ``None`` and
        label_encoding is specified, this is set to ``True``.
        If ``None`` and label_encoding is not specified, it defaults
        to ``False``.
    label_encoding : ``str``, optional (default=``None``)
        Label encoding to use when calculating span f1.
        Valid options are "BIO", "BIOUL", "IOB1", "BMES".
        Required if ``calculate_span_f1`` is true.
    labels_namespace : ``str``, optional (default=``labels``)
        This is needed to compute the SpanBasedF1Measure metric, if desired.
        Unless you did something unusual, the default value should be what you want.
    verbose_metrics : ``bool``, optional (default = False)
        If true, metrics will be returned per label class in addition
        to the overall statistics.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 predictor_dropout=0.0,
                 labels_namespace: str = "labels",
                 detect_namespace: str = "d_tags",
                 verbose_metrics: bool = False,
                 label_smoothing: float = 0.0,
                 confidence: float = 0.0,
                 del_confidence: float = 0.0,
                 max_len: int = 50,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 ) -> None:
        super(Seq2Tag2Seq, self).__init__(vocab, regularizer)

        self.label_namespaces = [labels_namespace,
                                 detect_namespace]
        self.text_field_embedder = text_field_embedder
        self.num_labels_classes = self.vocab.get_vocab_size(labels_namespace)
        self.num_detect_classes = self.vocab.get_vocab_size(detect_namespace)
        self.label_smoothing = label_smoothing
        self.confidence = confidence
        self.del_conf = del_confidence
        self.incorr_index = self.vocab.get_token_index("INCORRECT",
                                                       namespace=detect_namespace)

        self._verbose_metrics = verbose_metrics
        self.predictor_dropout = TimeDistributed(torch.nn.Dropout(predictor_dropout))

        self.tag_labels_projection_layer = TimeDistributed(
            Linear(text_field_embedder._token_embedders['bert'].get_output_dim(), self.num_labels_classes))

        self.tag_detect_projection_layer = TimeDistributed(
            Linear(text_field_embedder._token_embedders['bert'].get_output_dim(), self.num_detect_classes))

        self.metrics = {"accuracy": CategoricalAccuracy()}
        self.max_len = max_len

        initializer(self)

    def _convert(self, data):
        # print('data', data)
        all_class_probs = torch.zeros_like(data['class_probabilities_labels'])
        error_probs = torch.zeros_like(data['max_error_probability'])
        # print('sum(self.model_weights)', sum(self.model_weights))  # 1
        # for output, weight in zip(data, self.model_weights):
        #     all_class_probs += weight * output['class_probabilities_labels'] / sum(self.model_weights)
        #     error_probs += weight * output['max_error_probability'] / sum(self.model_weights)
        all_class_probs += data['class_probabilities_labels']
        error_probs += data['max_error_probability']


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

    def postprocess_batch(self, batch, all_probabilities, all_idxs, error_probs, need_tags=False):
        all_results = []
        all_tags = []
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
            # idxs = [2340, 2341, 2342, 2343, 2343, 2343, 2343, 2343, 2343]
            # print('tokens', tokens)
            # print('probabilities', probabilities)
            # print('idxs', idxs)

            length = min(len(tokens), self.max_len)
            edits = []
            tags = []

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
                tags.append(sugg_token)
                # print('sugg_token', sugg_token)
                action = self.get_token_action(token, i, probabilities[i],
                                               sugg_token)
                # print('action', action)
                if not action:
                    continue

                edits.append(action)
            all_results.append(get_target_sent_by_edits(tokens, edits))
            all_tags.append(tags)
            # print('after', get_target_sent_by_edits(tokens, edits))
            # print('all_results', all_results)
            # print('all_tags', all_tags)


        if need_tags == True:
            return all_tags, all_results
        else:
            return all_results


    @overrides
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor] = None,
                # labels: torch.LongTensor = None,
                # d_tags: torch.LongTensor = None,
                simplabels: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None,
                need_post_tag_batch = False,
                return_cls = False,
                logits_embedding = None) -> Dict[str, torch.Tensor]:

        # 只用于g_loss3的计算
        if logits_embedding is not None:
            # print('right')
            for embedder in self.text_field_embedder._token_embedders.values():
                # print('embedder', embedder)
                # exit()
                cls_embedding = embedder(logits_embedding=logits_embedding)
            return cls_embedding

        # print('tokens', tokens)
        encoded_text = self.text_field_embedder(tokens)

        # print('encoded_text.shape', encoded_text.shape)  # torch.Size([32, 40, 768])
        batch_size, sequence_length, _ = encoded_text.size()
        # print('tokens', tokens)

        # 对这里进行了修改
        # mask = get_text_field_mask(tokens)
        mask = (tokens['bert'] != 0).long()
        # print('mask', mask)
        logits_labels = self.tag_labels_projection_layer(self.predictor_dropout(encoded_text))
        # print('logits_labels', logits_labels.shape)
        logits_d = self.tag_detect_projection_layer(encoded_text)

        class_probabilities_labels = F.softmax(logits_labels, dim=-1).view(
            [batch_size, sequence_length, self.num_labels_classes])
        # print('class_probabilities_labels', class_probabilities_labels)
        class_probabilities_d = F.softmax(logits_d, dim=-1).view(
            [batch_size, sequence_length, self.num_detect_classes])
        # print('class_probabilities_d[:, :, self.incorr_index]', class_probabilities_d[:, :, self.incorr_index])
        error_probs = class_probabilities_d[:, :, self.incorr_index] * mask
        incorr_prob = torch.max(error_probs, dim=-1)[0]

        # probability_change = [self.confidence, self.del_conf] + [0] * (self.num_labels_classes - 2)
        # class_probabilities_labels += torch.FloatTensor(probability_change).repeat(
        #     (batch_size, sequence_length, 1)).to(class_probabilities_labels.device)

        output_dict = {"logits_labels": logits_labels,
                       "logits_d_tags": logits_d,
                       "class_probabilities_labels": class_probabilities_labels,
                       "class_probabilities_d_tags": class_probabilities_d,
                       "max_error_probability": incorr_prob}
        # if simplabels is not None :
        #     loss_labels = sequence_cross_entropy_with_logits(logits_labels, labels, mask,
        #                                                      label_smoothing=self.label_smoothing)
        #     # print('loss_labels', loss_labels)
        #     loss_d = sequence_cross_entropy_with_logits(logits_d, d_tags, mask)
        #     for metric in self.metrics.values():
        #         metric(logits_labels, labels, mask.float())
        #         metric(logits_d, d_tags, mask.float())
        #     output_dict["loss"] = loss_labels + loss_d

        if metadata is not None:
            output_dict["words"] = [x["words"] for x in metadata]
        # print("output_dict['logits_labels'].shape", output_dict['logits_labels'].shape)  # torch.Size([32, 40, 5002])
        # print("output_dict['logits_d_tags'].shape", output_dict['logits_d_tags'].shape)  # torch.Size([32, 40, 4])
        # print("output_dict['class_probabilities_labels'].shape", output_dict['class_probabilities_labels'].shape)  # torch.Size([32, 40, 5002])
        # print("output_dict['class_probabilities_d_tags'].shape", output_dict['class_probabilities_d_tags'].shape)  # torch.Size([32, 40, 4])
        # print("output_dict['max_error_probability'].shape", output_dict['max_error_probability'].shape)  # torch.Size([32])
        # print("output_dict['words'].shape", output_dict['words'].shape)  # list

        if need_post_tag_batch:
            # trans probabilities into tag_ids
            probabilities, idxs, error_probs = self._convert(output_dict)

            # trans tag_ids to word in sentences
            orig_batch = [i[1:] for i in output_dict["words"]]
            # orig_batch = output_dict["words"]

            # print('here')
            # print('orig_batch', orig_batch)
            # print('probabilities', probabilities)
            # print('idxs', idxs)
            # print('error_probs', error_probs)
            tags, post_tag_batch = self.postprocess_batch(orig_batch, probabilities,
                                            idxs, error_probs, need_tags=True)
        else:
            tags = None
            post_tag_batch = None
        # print('post_tag_batch', post_tag_batch)
        if return_cls:
            # cls_embedding = self.text_field_embedder(tokens, return_cls=True)
            # print('tokens', tokens)
            for embedder in self.text_field_embedder._token_embedders.values():
                cls_embedding = embedder(tokens['bert'], return_cls=return_cls)
            return output_dict, post_tag_batch, tags, cls_embedding
        else:
            return output_dict, post_tag_batch, tags

    # @overrides
    # def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    #     """
    #     Does a simple position-wise argmax over each token, converts indices to string labels, and
    #     adds a ``"tags"`` key to the dictionary with the result.
    #     """
    #     for label_namespace in self.label_namespaces:
    #         all_predictions = output_dict[f'class_probabilities_{label_namespace}']
    #         all_predictions = all_predictions.cpu().data.numpy()
    #         if all_predictions.ndim == 3:
    #             predictions_list = [all_predictions[i] for i in range(all_predictions.shape[0])]
    #         else:
    #             predictions_list = [all_predictions]
    #         all_tags = []
    #
    #         for predictions in predictions_list:
    #             argmax_indices = numpy.argmax(predictions, axis=-1)
    #             tags = [self.vocab.get_token_from_index(x, namespace=label_namespace)
    #                     for x in argmax_indices]
    #             all_tags.append(tags)
    #         output_dict[f'{label_namespace}'] = all_tags
    #     return output_dict
    #
    # @overrides
    # def get_metrics(self, reset: bool = False) -> Dict[str, float]:
    #     metrics_to_return = {metric_name: metric.get_metric(reset) for
    #                          metric_name, metric in self.metrics.items()}
    #     return metrics_to_return
