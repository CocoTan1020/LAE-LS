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


@Model.register("seq2tag_pipeline")
class Seq2Tag_pineline(Model):
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 predictor_dropout=0.0,
                 # labels_namespace: str = "labels",
                 # detect_namespace: str = "d_tags",
                 pipelinetag_namespace: str = "pipeline_tags",
                 verbose_metrics: bool = False,
                 label_smoothing: float = 0.0,
                 confidence: float = 0.0,
                 del_confidence: float = 0.0,
                 max_len: int = 50,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 ) -> None:
        super(Seq2Tag_pineline, self).__init__(vocab, regularizer)

        self.label_namespaces = [pipelinetag_namespace]
        self.text_field_embedder = text_field_embedder
        self.num_labels_classes = self.vocab.get_vocab_size(pipelinetag_namespace)
        self.label_smoothing = label_smoothing
        self.confidence = confidence
        self.del_conf = del_confidence

        self._verbose_metrics = verbose_metrics
        self.predictor_dropout = TimeDistributed(torch.nn.Dropout(predictor_dropout))

        self.tag_labels_projection_layer = TimeDistributed(
            Linear(text_field_embedder._token_embedders['bert'].get_output_dim(), self.num_labels_classes))
        # self.metrics = {"accuracy": CategoricalAccuracy()}
        self.max_len = max_len

        initializer(self)

    def _convert(self, data):
        # print('data', data)
        all_class_probs = torch.zeros_like(data['class_probabilities_labels'])
        # print('sum(self.model_weights)', sum(self.model_weights))  # 1
        # for output, weight in zip(data, self.model_weights):
        #     all_class_probs += weight * output['class_probabilities_labels'] / sum(self.model_weights)
        #     error_probs += weight * output['max_error_probability'] / sum(self.model_weights)
        all_class_probs += data['class_probabilities_labels']


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
        return probs, idx

    def postprocess_batch(self, batch, all_probabilities, all_idxs, need_tags=False):
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

        if need_tags == True:
            return all_tags, all_results
        else:
            return all_results

    def postprocess_batch_for_test(self, batch, all_probabilities, all_idxs, need_tags=False):
        # 只用于inference部分
        # 与postprocess_batch方法的区别：
        # 添加了超参数控制（阈值等）
        all_results = []
        all_tags = []
        for tokens, probabilities, idxs in zip(batch, all_probabilities, all_idxs):
            tags = []
            results = []
            # print('probabilities', probabilities)
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
                logits_embedding = None,
                for_test = False,
                labels = None,
                ) -> Dict[str, torch.Tensor]:

        # print('tokens', tokens)

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
        # print('logits_labels', logits_labels.shape)  # torch.Size([32, 30, 2])

        class_probabilities_labels = F.softmax(logits_labels, dim=-1).view(
            [batch_size, sequence_length, self.num_labels_classes])
        # print('class_probabilities_labels', class_probabilities_labels)

        output_dict = {"logits_labels": logits_labels,
                       "class_probabilities_labels": class_probabilities_labels}
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
            probabilities, idxs = self._convert(output_dict)
            # print('probabilities', probabilities)
            # print('idxs', idxs)

            # trans tag_ids to word in sentences
            orig_batch = [i for i in tokens['bert']]
            # print('orig_batch', orig_batch)


            # print('here')
            # print('orig_batch', orig_batch)
            # print('probabilities', probabilities)
            # print('idxs', idxs)
            # print('error_probs', error_probs)
            if for_test:
                tags, post_tag_batch = self.postprocess_batch_for_test(orig_batch, probabilities, idxs, need_tags=True)
            else:
                tags, post_tag_batch = self.postprocess_batch(orig_batch, probabilities, idxs, need_tags=True)
        else:
            tags = None
            post_tag_batch = None
        # print('post_tag_batch', post_tag_batch)
        if return_cls:
            # cls_embedding = self.text_field_embedder(tokens, return_cls=True)
            # print('tokens', tokens)
            for embedder in self.text_field_embedder._token_embedders.values():
                cls_embedding = embedder(tokens['bert'], return_cls=return_cls)
            return output_dict, post_tag_batch, tags, cls_embedding, orig_batch
        elif for_test:
            return output_dict, orig_batch, post_tag_batch, tags
        else:
            return output_dict, post_tag_batch, tags


