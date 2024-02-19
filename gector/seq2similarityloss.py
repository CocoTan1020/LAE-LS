
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


@Model.register("seq2similarityloss")
class Seq2Similarityloss(Model):
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 verbose_metrics: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(Seq2Similarityloss, self).__init__(vocab, regularizer)
        self.text_field_embedder = text_field_embedder
        self._verbose_metrics = verbose_metrics
        self.cosineembeddingloss = nn.CosineEmbeddingLoss()
        initializer(self)

    @overrides
    def forward(self,
                tokens1: Dict[str, torch.LongTensor],
                tokens2: Dict[str, torch.LongTensor]
                ) -> Dict[str, torch.Tensor]:

        device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        for k in tokens1.keys():
            tokens1[k].to(device)
        for k in tokens2.keys():
            tokens2[k].to(device)
        cls_embedding1 = self.text_field_embedder(tokens1).to(device)
        cls_embedding2 = self.text_field_embedder(tokens2).to(device)
        # print('cls_embedding1.shape', cls_embedding1.shape)  # (batch_size, 768)
        batch_size, _ = cls_embedding1.size()
        target = torch.ones(batch_size)
        loss = self.cosineembeddingloss(cls_embedding1, cls_embedding2, target)
        # print(loss)
        output_dict = {"loss": loss}
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = {metric_name: metric.get_metric(reset) for
                             metric_name, metric in self.metrics.items()}
        # print('dddddd')
        # for metric_name, metric in self.metrics.items():
        #     print(metric.get_metric())
        return metrics_to_return


class Remainloss(nn.Module):
    def __init__(self):
        super().__init__()
        self._criterion = nn.CrossEntropyLoss(reduction='none')
    def forward(self, logits, targets):

        # print('logits', logits.shape)  # torch.Size([32, 30, 30522])
        # print('targets', targets.shape)  # torch.Size([32, 30])
        vocab_size = logits.size(-1)
        tgts = targets.contiguous().view(-1) # tgts: (N)
        # print('tgts', tgts.shape)
        outs = logits.contiguous().view(-1, vocab_size) # outs: (N, V)
        # print('outs', outs.shape)
        loss = self._criterion(outs, tgts) # [N]
        # print('loss', logits)
        output_dict = {"loss": loss.mean()}
        return output_dict

class Semanticloss(nn.Module):
    def __init__(self):
        super().__init__()
        self._criterion = nn.CosineEmbeddingLoss()
    def forward(self, logits, targets, device):

        # print('logits', logits.shape)  # torch.Size([32, 30, 30522])
        # print('targets', targets.shape)  # torch.Size([32, 30])
        batch_size = logits.size()[0]
        target = torch.ones(batch_size).to(device)
        loss = self._criterion(logits, targets, target)
        # print(loss)
        output_dict = {"loss": loss}
        return output_dict


class MarginLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(MarginLoss, self).__init__()
        self.margin = margin

    def forward(self, y_pred):
        # 计算预测值与0.5之间的绝对差值
        margin_penalty = torch.abs(y_pred - 0.5) - self.margin
        # 使用均方误差作为样本分类的损失
        classification_loss = torch.mean((y_pred - 0.5) ** 2)
        # 添加margin惩罚项
        margin_loss = torch.clamp(margin_penalty, min=0.0)
        # 最终的loss由样本分类损失和margin惩罚项组成
        final_loss = classification_loss + margin_loss.mean()

        return final_loss

class ContrastLoss(nn.Module):
    def __init__(self, scale=1.0):
        super(ContrastLoss, self).__init__()
        self.scale = scale

    def forward(self, logits_before, logits_after):
        loss = torch.exp(-self.scale * (logits_after - logits_before))
        return loss.mean()
