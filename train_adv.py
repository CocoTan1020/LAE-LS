# coding=utf-8
import argparse
import os
from random import seed

import torch
from allennlp.data.iterators import BucketIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder

from gector.bert_token_embedder import PretrainedBertEmbedder
from gector.datareader import Seq2SimpLabelDatasetReader, Seq2LabelsDatasetReader, GeneratorDatasetReader, GeneratorLLMDatasetReader
from gector.seq2simplabel_model import Seq2SimpLabel
from gector.seq2similarityloss import Seq2Similarityloss
from gector.trainer import Trainer_Discriminator, Trainer_Generator, Trainer_Adversarial, Trainer_Adversarial_pipeline_llm
from gector.tokenizer_indexer import PretrainedBertIndexer
from gector.seq2tag2seq_model import Seq2Tag2Seq
from gector.seq2tag_pipeline_model import Seq2Tag_pineline
from utils.helpers import get_weights_name
import sys

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

log = Logger('log_adv.txt')
sys.stdout = log

def fix_seed():
    torch.manual_seed(1)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    seed(43)


def get_token_indexers(model_name, max_pieces_per_token=5, lowercase_tokens=True, special_tokens_fix=0, pre_pretrained_model_path=None):
    bert_token_indexer = PretrainedBertIndexer(
        pretrained_model=model_name,
        max_pieces_per_token=max_pieces_per_token,
        do_lowercase=lowercase_tokens,
        special_tokens_fix=special_tokens_fix,
        pre_pretrained_model_path=pre_pretrained_model_path
    )
    return {'bert': bert_token_indexer}


def get_token_embedders_for_generator(model_name, tune_bert=False, special_tokens_fix=0, pre_pretrained_model_path=None):
    take_grads = True if tune_bert > 0 else False
    bert_token_emb = PretrainedBertEmbedder(
        pretrained_model=model_name,
        top_layer_only=True, requires_grad=take_grads,
        special_tokens_fix=special_tokens_fix,
        pre_pretrained_model_path=pre_pretrained_model_path,
        for_sequence=False
    )

    token_embedders = {'bert': bert_token_emb}
    embedder_to_indexer_map = {"bert": ["bert", "bert-offsets"]}

    text_filed_emd = BasicTextFieldEmbedder(token_embedders=token_embedders,
                                            embedder_to_indexer_map=embedder_to_indexer_map,
                                            allow_unmatched_keys=True)
    return text_filed_emd


def get_token_embedders_for_discriminator(model_name, tune_bert=False,
                                          special_tokens_fix=0, pre_pretrained_model_path=None):
    take_grads = True if tune_bert > 0 else False
    bert_token_emb = PretrainedBertEmbedder(
        pretrained_model=model_name,
        top_layer_only=True, requires_grad=take_grads,
        special_tokens_fix=special_tokens_fix,
        pre_pretrained_model_path=pre_pretrained_model_path,
        for_sequence=True,  # difference from get_token_embedders_for_generator
    )

    token_embedders = {'bert': bert_token_emb}
    embedder_to_indexer_map = {"bert": ["bert", "bert-offsets"]}

    text_filed_emd = BasicTextFieldEmbedder(token_embedders=token_embedders,
                                            embedder_to_indexer_map=embedder_to_indexer_map,
                                            allow_unmatched_keys=True)
    return text_filed_emd


def get_data_reader_for_generator(model_name, max_len, skip_correct=False, skip_complex=0,
                    test_mode=False, tag_strategy="keep_one",
                    broken_dot_strategy="keep", lowercase_tokens=True,
                    max_pieces_per_token=3, tn_prob=0, tp_prob=1, special_tokens_fix=0, pre_pretrained_model_path=None):
    token_indexers = get_token_indexers(model_name,
                                        max_pieces_per_token=max_pieces_per_token,
                                        lowercase_tokens=lowercase_tokens,
                                        special_tokens_fix=special_tokens_fix,
                                        pre_pretrained_model_path=pre_pretrained_model_path
                                        )
    reader = GeneratorLLMDatasetReader(token_indexers=token_indexers,
                                     max_len=max_len,
                                     skip_correct=skip_correct,
                                     skip_complex=skip_complex,
                                     test_mode=test_mode,
                                     tag_strategy=tag_strategy,
                                     broken_dot_strategy=broken_dot_strategy,
                                     lazy=True,
                                     tn_prob=tn_prob,
                                     tp_prob=tp_prob)
    return reader


def get_data_reader_for_discriminator(model_name, max_len, skip_correct=False, skip_complex=0,
                    test_mode=False, tag_strategy="keep_one",
                    broken_dot_strategy="keep", lowercase_tokens=True,
                    max_pieces_per_token=3, tn_prob=0, tp_prob=1, special_tokens_fix=0, pre_pretrained_model_path=None):
    token_indexers = get_token_indexers(model_name,
                                        max_pieces_per_token=max_pieces_per_token,
                                        lowercase_tokens=lowercase_tokens,
                                        special_tokens_fix=special_tokens_fix,
                                        pre_pretrained_model_path=pre_pretrained_model_path
                                        )
    reader = Seq2SimpLabelDatasetReader(token_indexers=token_indexers,
                                     max_len=max_len,
                                     skip_correct=skip_correct,
                                     skip_complex=skip_complex,
                                     test_mode=test_mode,
                                     tag_strategy=tag_strategy,
                                     broken_dot_strategy=broken_dot_strategy,
                                     lazy=True,
                                     tn_prob=tn_prob,
                                     tp_prob=tp_prob)
    return reader


def get_genarator_model(model_name, vocab, tune_bert=False,
              predictor_dropout=0,
              special_tokens_fix=0,
              pre_pretrained_model_path=None):
    token_embs = get_token_embedders_for_generator(model_name, tune_bert=tune_bert, special_tokens_fix=special_tokens_fix, pre_pretrained_model_path=pre_pretrained_model_path)
    model = Seq2Tag_pineline(vocab=vocab,
                          predictor_dropout=predictor_dropout,
                          text_field_embedder=token_embs)
    return model


def get_discriminator_model(model_name, vocab, tune_bert=False,
              predictor_dropout=0,
              special_tokens_fix=0,
              pre_pretrained_model_path=None):
    token_embs = get_token_embedders_for_discriminator(model_name, tune_bert=tune_bert, special_tokens_fix=special_tokens_fix, pre_pretrained_model_path=pre_pretrained_model_path)
    model = Seq2SimpLabel(vocab=vocab,
                          predictor_dropout=predictor_dropout,
                          text_field_embedder=token_embs)
    return model

def get_similarity_model(model_name, vocab, tune_bert=False,
              predictor_dropout=0,
              special_tokens_fix=0,
              pre_pretrained_model_path=None):
    token_embs = get_token_embedders_for_discriminator(model_name, tune_bert=tune_bert, special_tokens_fix=special_tokens_fix, pre_pretrained_model_path=pre_pretrained_model_path)
    model = Seq2Similarityloss(vocab=vocab,
                          text_field_embedder=token_embs)
    return model


def main(args):
    print('args: ', args)
    fix_seed()
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)

    weights_name = get_weights_name(args.transformer_model, args.lowercase_tokens)
    # print('weights_name', weights_name)
    # read datasets
    reader = get_data_reader_for_generator(weights_name, args.max_len, skip_correct=bool(args.skip_correct),
                             skip_complex=args.skip_complex,
                             test_mode=False,
                             tag_strategy=args.tag_strategy,
                             lowercase_tokens=args.lowercase_tokens,
                             max_pieces_per_token=args.pieces_per_token,
                             tn_prob=args.tn_prob,
                             tp_prob=args.tp_prob,
                             special_tokens_fix=args.special_tokens_fix,
                             pre_pretrained_model_path=args.pre_pretrained_model_path)
    reader_discriminator = get_data_reader_for_discriminator(weights_name, args.max_len, skip_correct=bool(args.skip_correct),
                                           skip_complex=args.skip_complex,
                                           test_mode=False,
                                           tag_strategy=args.tag_strategy,
                                           lowercase_tokens=args.lowercase_tokens,
                                           max_pieces_per_token=args.pieces_per_token,
                                           tn_prob=args.tn_prob,
                                           tp_prob=args.tp_prob,
                                           special_tokens_fix=args.special_tokens_fix,
                                           pre_pretrained_model_path=args.pre_pretrained_model_path)

    train_source = reader.read(args.train_source)
    train_target = reader.read(args.train_target)
    print('train_source', train_source)
    for field in train_source:
        print(field)
        break
    print('train_target', train_target)
    for field in train_target:
        print(field)
        break
    # (tokens, metadata, simplabel)
    # tokens: {'bert': tensor(), 'bert-offsets': tensor(), 'mask': tensor()}
    # metadata: [{'words':[...]}, ...]

    valid_source = reader.read(args.valid_source)
    valid_target = reader.read(args.valid_target)
    # print('valid_source', valid_source)
    # print('valid_target', valid_target)
    print('valid_source', valid_source)
    for field in valid_source:
        print(field)
        break
    print('valid_target', valid_target)
    for field in valid_target:
        print(field)
        break

    test_source = reader.read(args.test_source)
    test_target = reader.read(args.test_target)


    # default_tokens = [DEFAULT_OOV_TOKEN, DEFAULT_PADDING_TOKEN]
    # namespaces = ['simplabel']
    # tokens_to_add = {x: default_tokens for x in namespaces}
    # build vocab
    if args.vocab_path:
        vocab = Vocabulary.from_files(args.vocab_path)
    else:
        print('Vocabulary should be provided. [Vocabulary.from_instances()] to be completed.')
    vocab.save_to_files(os.path.join(args.model_dir, 'vocabulary'))
    print('vocab', vocab)

    print("Data is loaded")
    generator = get_genarator_model(weights_name, vocab,
                      tune_bert=args.tune_bert,
                      predictor_dropout=args.predictor_dropout,
                      special_tokens_fix=args.special_tokens_fix,
                      pre_pretrained_model_path=args.pre_pretrained_model_path)
    print('generator', generator)
    discriminator = get_discriminator_model(weights_name, vocab,
                                    tune_bert=args.tune_bert,
                                    predictor_dropout=args.predictor_dropout,
                                    special_tokens_fix=args.special_tokens_fix,
                                    pre_pretrained_model_path=args.pre_pretrained_model_path)
    print('discriminator', discriminator)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        cuda_device = 0
    else:
        cuda_device = -1

    if args.pretrain_discriminator:
        print('Load pretrain_discriminator from ', end='')
        print(str(os.path.join(args.pretrain_discriminator_folder, args.pretrain_discriminator + '.th')))
        discriminator.load_state_dict(
            torch.load(os.path.join(args.pretrain_discriminator_folder, args.pretrain_discriminator + '.th'), map_location=torch.device('cpu')),
            strict=True,
        )
        # print(discriminator)
        # exit()

    if args.pretrain_generator:
        print('Load pretrain_generator from ', end='')
        print(str(os.path.join(args.pretrain_generator_folder, args.pretrain_generator + '.th')))
        generator.load_state_dict(
            torch.load(os.path.join(args.pretrain_generator_folder, args.pretrain_generator + '.th')),
            strict=True,
        )

    generator = generator.to(device)
    discriminator = discriminator.to(device)


    print("Model is set")

    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=args.lr)
    scheduler_generator = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_generator, factor=0.1, patience=10)
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=args.lr)
    scheduler_discriminator = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_discriminator, factor=0.1, patience=10)

    instances_per_epoch = None if not args.updates_per_epoch else \
        int(args.updates_per_epoch * args.batch_size * args.accumulation_size)
    iterator = BucketIterator(batch_size=args.batch_size,
                              sorting_keys=[("tokens", "num_tokens")],
                              biggest_batch_first=True,
                              max_instances_in_memory=instances_per_epoch,
                              instances_per_epoch=instances_per_epoch,
                              )
    iterator.index_with(vocab)
    val_iterator = BucketIterator(batch_size=args.batch_size,
                                  sorting_keys=[("tokens", "num_tokens")],
                                  instances_per_epoch=None)
    val_iterator.index_with(vocab)

    trainer = Trainer_Adversarial_pipeline_llm(generator=generator,
                                  discriminator=discriminator,
                                  reader_discriminator=reader_discriminator,
                                  optimizer_generator=optimizer_generator,
                                  optimizer_discriminator=optimizer_discriminator,
                                  schedule_generatorr=scheduler_generator,
                                  scheduler_discriminator=scheduler_discriminator,
                                  iterator=iterator,
                                  validation_iterator=val_iterator,
                                  train_source=train_source,
                                  train_target=train_target,
                                  valid_source=valid_source,
                                  valid_target=valid_target,
                                  test_source=test_source,
                                  test_target=test_target,
                                  generator_update_batch=args.generator_update_batch,
                                  discriminator_update_batch=args.discriminator_update_batch,
                                  serialization_dir=args.model_dir,
                                  patience=args.patience,
                                  num_epochs=args.n_epoch,
                                  cuda_device=cuda_device,
                                  shuffle=True,
                                  g_loss1_hp=args.g_loss1_hp,
                                  g_loss2_hp=args.g_loss2_hp,
                                  g_loss3_hp=args.g_loss3_hp,
                                  g_loss4_hp=args.g_loss4_hp,
                                  g_loss5_hp=args.g_loss5_hp,
                                  g_loss6_hp=args.g_loss6_hp,
                                  g_loss7_hp=args.g_loss7_hp,
                                  g_loss8_hp=args.g_loss8_hp,
                                  d_loss1_hp=args.d_loss1_hp,
                                  d_loss2_hp=args.d_loss2_hp,
                                  d_loss3_hp=args.d_loss3_hp,
                                  accumulated_batch_count=args.accumulation_size,
                                  cold_step_count=args.cold_steps_count,
                                  cold_lr=args.cold_lr,
                                  train_temp_path=args.train_temp_path,
                                  cuda_verbose_step=int(args.cuda_verbose_steps)
                                  if args.cuda_verbose_steps else None
                                  )
    print("Start training")
    trainer.train()

    # Here's how to save the model.
    # out_model = os.path.join(args.model_dir, 'generator.th')
    # with open(out_model, 'wb') as f:
    #     torch.save(generator.state_dict(), f)
    # print("Model-generator is dumped")
    #
    # out_model = os.path.join(args.model_dir, 'discriminator.th')
    # with open(out_model, 'wb') as f:
    #     torch.save(discriminator.state_dict(), f)
    # print("Model-discriminator is dumped")


if __name__ == '__main__':
    # read parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_source',
                        help='Path to the train source corpus data',
                        default='data/wikismall/llm_cwi/train_gpt-3.5-turbo-organized.json')
    parser.add_argument('--train_target',
                        help='Path to the train target corpus data',
                        default='data/wikismall/train_dst.txt')
    parser.add_argument('--valid_source',
                        help='Path to the valid target corpus data',
                        default='data/wikismall/llm_cwi/valid_gpt-3.5-turbo-organized.json')
    parser.add_argument('--valid_target',
                        help='Path to the valid target corpus data',
                        default='data/wikismall/valid_dst.txt')
    parser.add_argument('--test_source',
                        help='Path to the valid target corpus data',
                        default='data/wikismall/llm_cwi/test_gpt-3.5-turbo-organized.json')
    parser.add_argument('--test_target',
                        help='Path to the valid target corpus data',
                        default='data/wikismall/test_dst.txt')
    parser.add_argument('--model_dir',
                        help='Path to the model dir',
                        default='saved_dict_adv')
    parser.add_argument('--vocab_path',
                        help='Path to the model vocabulary directory.'
                             'If not set then build vocab from data',
                        default='saved_dict_adv/vocabulary')
    parser.add_argument('--batch_size',
                        type=int,
                        help='The size of the batch.',
                        default=4)
    parser.add_argument('--max_len',
                        type=int,
                        help='The max sentence length'
                             '(all longer will be truncated)',
                        default=50)
    parser.add_argument('--target_vocab_size',
                        type=int,
                        help='The size of target vocabularies.',
                        default=1000)
    parser.add_argument('--n_epoch',
                        type=int,
                        help='The number of epoch for training model.',
                        default=30)
    parser.add_argument('--patience',
                        type=int,
                        help='The number of epoch with any improvements'
                             ' on validation set.',
                        default=30000)
    parser.add_argument('--skip_correct',
                        type=int,
                        help='If set than correct sentences will be skipped '
                             'by data reader.',
                        default=1)
    parser.add_argument('--skip_complex',
                        type=int,
                        help='If set than complex corrections will be skipped '
                             'by data reader.',
                        choices=[0, 1, 2, 3, 4, 5],
                        default=0)
    parser.add_argument('--tune_bert',
                        type=int,
                        help='If more then 0 then fine tune bert.',
                        default=1)
    parser.add_argument('--tag_strategy',
                        choices=['keep_one', 'merge_all'],
                        help='The type of the data reader behaviour.',
                        default='keep_one')
    parser.add_argument('--accumulation_size',
                        type=int,
                        help='How many batches do you want accumulate.',
                        default=4)
    parser.add_argument('--lr',
                        type=float,
                        help='Set initial learning rate.',
                        default=1e-5)
    parser.add_argument('--cold_steps_count',
                        type=int,
                        help='Whether to train only classifier layers first.',
                        default=4)
    parser.add_argument('--cold_lr',
                        type=float,
                        help='Learning rate during cold_steps.',
                        default=1e-3)
    parser.add_argument('--predictor_dropout',
                        type=float,
                        help='The value of dropout for predictor.',
                        default=0.0)
    parser.add_argument('--lowercase_tokens',
                        type=int,
                        help='Whether to lowercase tokens.',
                        default=0)
    parser.add_argument('--pieces_per_token',
                        type=int,
                        help='The max number for pieces per token.',
                        default=5)
    parser.add_argument('--cuda_verbose_steps',
                        help='Number of steps after which CUDA memory information is printed. '
                             'Makes sense for local testing. Usually about 1000.',
                        default=None)
    parser.add_argument('--label_smoothing',
                        type=float,
                        help='The value of parameter alpha for label smoothing.',
                        default=0.0)
    parser.add_argument('--tn_prob',
                        type=float,
                        help='The probability to take TN from data.',
                        default=0)
    parser.add_argument('--tp_prob',
                        type=float,
                        help='The probability to take TP from data.',
                        default=1)
    parser.add_argument('--updates_per_epoch',
                        type=int,
                        help='If set then each epoch will contain the exact amount of updates.',
                        default=0)
    parser.add_argument('--pretrain_discriminator_folder',
                        help='The name of the pretrain_discriminator folder.',
                        default='finished_trained_models/pretrained_discriminator_bert_without_ls')
    parser.add_argument('--pretrain_discriminator',
                        help='The name of the pretrain weights in pretrain_discriminator_folder param.',
                        default='model_state_epoch_29')
    parser.add_argument('--pretrain_generator_folder',
                        help='The name of the pretrain_generator folder.')
    parser.add_argument('--pretrain_generator',
                        help='The name of the pretrain weights in pretrain_generator_folder param.',
                        default='')
    parser.add_argument('--transformer_model',
                        choices=['bert', 'distilbert', 'gpt2', 'roberta', 'transformerxl', 'xlnet', 'albert',
                                 'bert-large', 'roberta-large', 'xlnet-large'],
                        help='Name of the transformer model.',
                        default='bert')
    parser.add_argument('--special_tokens_fix',
                        type=int,
                        help='Whether to fix problem with [CLS], [SEP] tokens tokenization.',
                        default=0)
    parser.add_argument('--pre_pretrained_model_path',
                        help='Path to the pretrained model file related to --transformer_model. (None for auto downloading from huggingface web.)',
                        default='pre_pretrained_models/bert')
    parser.add_argument('--generator_update_batch',
                        type=int,
                        help='Update generator every batch once.',
                        default=2)
    parser.add_argument('--discriminator_update_batch',
                        type=int,
                        help='Update discriminator every batch once.',
                        default=100000000000)

    parser.add_argument('--g_loss1_hp',
                        type=float,
                        default=0)
    parser.add_argument('--g_loss2_hp',
                        type=float,
                        default=0)
    parser.add_argument('--g_loss3_hp',
                        type=float,
                        default=1)
    parser.add_argument('--g_loss4_hp',
                        type=float,
                        default=0)
    parser.add_argument('--g_loss5_hp',
                        type=float,
                        default=0)
    parser.add_argument('--g_loss6_hp',
                        type=float,
                        default=0)
    parser.add_argument('--g_loss7_hp',
                        type=float,
                        default=0)
    parser.add_argument('--g_loss8_hp',
                        type=float,
                        default=0)
    parser.add_argument('--d_loss1_hp',
                        type=float,
                        default=1)
    parser.add_argument('--d_loss2_hp',
                        type=float,
                        default=0)
    parser.add_argument('--d_loss3_hp',
                        type=float,
                        default=0)
    parser.add_argument('--train_temp_path',
                        default='train_temp2')

    args = parser.parse_args()

    main(args)



