"""
Discriminator model.
Predict simp_labels for every sequences.
simple; not simple
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
import torch.nn as nn
from random import seed

def fix_seed():
    torch.manual_seed(1)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    seed(43)
fix_seed()


def freeze(model):
    for param in model.parameters():
        param.requires_grad = False

@Model.register("seq2simplabel")
class Seq2SimpLabel(Model):
    """
    This ``Seq2SimpLabel`` simply encodes a sequence of text with a stacked ``Seq2SeqEncoder``, then
    predicts a simp_label for the sequence.

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
                 verbose_metrics: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(Seq2SimpLabel, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_simplabel_classes = 2

        self._verbose_metrics = verbose_metrics
        self.predictor_dropout = torch.nn.Dropout(predictor_dropout)

        self.simplabel_projection_layer = Linear(self.text_field_embedder._token_embedders['bert'].get_output_dim(), self.num_simplabel_classes)
        self.crossentropyloss = nn.CrossEntropyLoss()
        self.metrics = {"accuracy": CategoricalAccuracy()}

        initializer(self)
        # freeze(self)

    @overrides
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor] = None,
                simplabels: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None,
                logits_embedding: torch.float = None,
                mask = None,
                labels = None) -> Dict[str, torch.Tensor]:
        # 只用于对抗部分的generator的loss1使用
        if logits_embedding is not None:
            # print('right')
            for embedder in self.text_field_embedder._token_embedders.values():
                # print('embedder', self.text_field_embedder._token_embedders)
                # print('embedder', self.text_field_embedder)
                # exit()
                cls_embedding = embedder(logits_embedding=logits_embedding, attention_mask=mask)
        else:
            cls_embedding = self.text_field_embedder(tokens)
        # print('cls_embedding.shape', cls_embedding.shape)  # (batch_size, 768)
        batch_size, _ = cls_embedding.size()
        # mask = get_text_field_mask(tokens)
        logits_simplabel = self.simplabel_projection_layer(self.predictor_dropout(cls_embedding))
        # print('logits_simplabel.shape', logits_simplabel.shape)  # torch.Size([32, 2])
        # print('logits_simplabel', logits_simplabel)
        class_probabilities_simplabel = F.softmax(logits_simplabel, dim=-1).view(
            [batch_size, self.num_simplabel_classes])
        # print('class_probabilities_simplabel.shape', class_probabilities_simplabel.shape)  # torch.Size([32, 2])
        # print('class_probabilities_simplabel', class_probabilities_simplabel)
        output_dict = {"logits_simplabel": logits_simplabel,
                       "class_probabilities_simplabel": class_probabilities_simplabel}

        if simplabels is not None:
            loss_simplabel = self.crossentropyloss(logits_simplabel, simplabels)
            # print('loss_simplabel', loss_simplabel)
            for metric in self.metrics.values():
                metric(logits_simplabel, simplabels)
            output_dict["loss"] = loss_simplabel
            y_hat = torch.argmax(class_probabilities_simplabel, dim=1)
            # print('y_hat', y_hat)
            accuracy = torch.eq(y_hat, simplabels.squeeze(dim=-1)).float().mean()
            output_dict['accuracy'] = accuracy.detach().cpu().numpy()

        if metadata is not None:
            output_dict["words"] = [x["words"] for x in metadata]

        # output_dict["last_layer_cls_attention"]
        # print('output_dict', output_dict)
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = {metric_name: metric.get_metric(reset) for
                             metric_name, metric in self.metrics.items()}
        # print('dddddd')
        # for metric_name, metric in self.metrics.items():
        #     print(metric.get_metric())
        return metrics_to_return






