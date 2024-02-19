"""Tweaked AllenNLP dataset reader."""
import json
import logging
import re
from random import random
from typing import Dict, List

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField, MetadataField, Field, LabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from overrides import overrides

from utils.helpers import SEQ_DELIMETERS, START_TOKEN

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


# For gector training only
#   input: words with tags
#   output: tokens, metadata, labels, d_tages
@DatasetReader.register("seq2labels_datareader")
class Seq2LabelsDatasetReader(DatasetReader):
    """
    Reads instances from a pretokenised file where each line is in the following format:

    WORD###TAG [TAB] WORD###TAG [TAB] ..... \n

    and converts it into a ``Dataset`` suitable for sequence tagging. You can also specify
    alternative delimiters in the constructor.

    Parameters
    ----------
    delimiters: ``dict``
        The dcitionary with all delimeters.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
        Note that the `output` tags will always correspond to single token IDs based on how they
        are pre-tokenised in the data file.
    max_len: if set than will truncate long sentences
    """
    # fix broken sentences mostly in Lang8
    BROKEN_SENTENCES_REGEXP = re.compile(r'\.[a-zA-RT-Z]')

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 delimeters: dict = SEQ_DELIMETERS,
                 skip_correct: bool = False,
                 skip_complex: int = 0,
                 lazy: bool = False,
                 max_len: int = None,
                 test_mode: bool = False,
                 tag_strategy: str = "keep_one",
                 tn_prob: float = 0,
                 tp_prob: float = 0,
                 broken_dot_strategy: str = "keep",
                 only_keep_delete_replace = False,
                 labels_vocab_path=None) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._delimeters = delimeters
        self._max_len = max_len
        self._skip_correct = skip_correct
        self._skip_complex = skip_complex
        self._tag_strategy = tag_strategy
        self._broken_dot_strategy = broken_dot_strategy
        self._test_mode = test_mode
        self._tn_prob = tn_prob
        self._tp_prob = tp_prob
        self._only_keep_delete_replace = only_keep_delete_replace
        self._labels_vocab_path = labels_vocab_path
        if self._labels_vocab_path is not None:
            f = open(self._labels_vocab_path, encoding='utf-8')
            self.labels_vocab = [i.replace("\n", "") for i in f.readlines()]

    @overrides
    def _read(self, file_path):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        with open(file_path, "r", encoding='UTF-8') as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line in data_file:
                line = line.strip("\n")
                # skip blank and broken lines
                if not line or (not self._test_mode and self._broken_dot_strategy == 'skip'
                                and self.BROKEN_SENTENCES_REGEXP.search(line) is not None):
                    continue

                tokens_and_tags = [pair.rsplit(self._delimeters['labels'], 1)
                                   for pair in line.split(self._delimeters['tokens'])]
                # print('tokens_and_tags', tokens_and_tags)
                try:
                    tokens = [Token(token) for token, tag in tokens_and_tags]
                    tags = [tag for token, tag in tokens_and_tags]
                except ValueError:
                    tokens = [Token(token[0]) for token in tokens_and_tags]
                    tags = None

                if tokens and tokens[0] != Token(START_TOKEN):
                    tokens = [Token(START_TOKEN)] + tokens
                # tokens = [Token('[CLS]')] + tokens

                words = [x.text for x in tokens]
                if self._max_len is not None:
                    tokens = tokens[:self._max_len]
                    tags = None if tags is None else tags[:self._max_len]

                # print('token', tokens)  # token [$START, Bowdon, is, a, city, in, Carroll, County, ,, Georgia, ,, United, States, .]
                # print('tags', tags)  # tags ['$KEEP', '$DELETE', '$DELETE', '$DELETE', '$DELETE', '$DELETE', '$DELETE', '$DELETE', '$DELETE', '$DELETE', '$DELETE', '$KEEP', '$KEEP', '$KEEP']
                # print('words', words)  # words ['$START', 'Bowdon', 'is', 'a', 'city', 'in', 'Carroll', 'County', ',', 'Georgia', ',', 'United', 'States', '.']
                instance = self.text_to_instance(tokens, tags, words)
                if instance:
                    yield instance

    def extract_tags(self, tags: List[str]):
        # print('tags', tags)
        op_del = self._delimeters['operations']
        # print('op_del', op_del)
        labels = [x.split(op_del) for x in tags]

        comlex_flag_dict = {}
        # get flags
        for i in range(5):
            idx = i + 1
            comlex_flag_dict[idx] = sum([len(x) > idx for x in labels])

        if self._tag_strategy == "keep_one":
            # get only first candidates for r_tags in right and the last for left
            _labels = [x[0] for x in labels]
            if self._only_keep_delete_replace == True:
                labels = []
                for l in _labels:
                    if l == '$KEEP' or l == '$DELETE':
                        labels.append(l)
                    elif l.startswith('$REPLACE'):
                        temp = l[9:].lower()
                        if temp in self.labels_vocab:
                            labels.append(temp)
                        else:
                            labels.append('$DELETE')
                    else:
                        labels.append('$KEEP')
        elif self._tag_strategy == "merge_all":
            # consider phrases as a words
            pass
        else:
            raise Exception("Incorrect tag strategy")

        detect_tags = ["CORRECT" if label == "$KEEP" else "INCORRECT" for label in labels]
        return labels, detect_tags, comlex_flag_dict

    def text_to_instance(self, tokens: List[Token], tags: List[str] = None,
                         words: List[str] = None) -> Instance:  # type: ignore
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.
        """
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        sequence = TextField(tokens, self._token_indexers)
        # print('sequence', sequence)
        fields["tokens"] = sequence
        fields["metadata"] = MetadataField({"words": words})
        if tags is not None:
            labels, detect_tags, complex_flag_dict = self.extract_tags(tags)

            if self._skip_complex and complex_flag_dict[self._skip_complex] > 0:
                return None
            rnd = random()
            # skip TN
            if self._skip_correct and all(x == "CORRECT" for x in detect_tags):
                if rnd > self._tn_prob:
                    return None
            # skip TP
            else:
                if rnd > self._tp_prob:
                    return None

            fields["labels"] = SequenceLabelField(labels, sequence,
                                                  label_namespace="labels")
            fields["d_tags"] = SequenceLabelField(detect_tags, sequence,
                                                  label_namespace="d_tags")
        # print('fields["tokens"]', fields["tokens"])
        # print('fields["metadata"]', fields["metadata"])
        # print('fields["labels"]', fields["labels"])
        # print('fields["d_tags"]', fields["d_tags"])
        return Instance(fields)


# For discriminator training only
#   input: sentence /t simplabel
#   output: tokens, metadata, simplabels
@DatasetReader.register("seq2simplabel_datareader")
class Seq2SimpLabelDatasetReader(DatasetReader):
    """
    Reads instances from a pretokenised file where each line is in the following format:

    WORD WORD WORD WORD WORD WORD WORD ... [TAB]simplabel\n

    and converts it into a ``Dataset`` suitable for simplabel prediction. You can also specify
    alternative delimiters in the constructor.

    Parameters
    ----------
    delimiters: ``dict``
        The dcitionary with all delimeters.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
        Note that the `output` tags will always correspond to single token IDs based on how they
        are pre-tokenised in the data file.
    max_len: if set than will truncate long sentences
    """
    # fix broken sentences mostly in Lang8
    BROKEN_SENTENCES_REGEXP = re.compile(r'\.[a-zA-RT-Z]')

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 delimeters: dict = SEQ_DELIMETERS,
                 skip_correct: bool = False,
                 skip_complex: int = 0,
                 lazy: bool = False,
                 max_len: int = None,
                 test_mode: bool = False,
                 tag_strategy: str = "keep_one",
                 tn_prob: float = 0,
                 tp_prob: float = 0,
                 broken_dot_strategy: str = "keep") -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._delimeters = delimeters
        self._max_len = max_len
        self._skip_correct = skip_correct
        self._skip_complex = skip_complex
        self._tag_strategy = tag_strategy
        self._broken_dot_strategy = broken_dot_strategy
        self._test_mode = test_mode
        self._tn_prob = tn_prob
        self._tp_prob = tp_prob

    @overrides
    def _read(self, file_path):
        # if `file_path` is a URL, redirect to the cache
        # file_path = cached_path(file_path)

        if type(file_path) == type([]):
            for line, sim in zip(file_path[0], file_path[1]):
                tokens = [Token(i) for i in line]
                simplabel = [sim]

                # if tokens and tokens[0] != Token(START_TOKEN):
                #     tokens = [Token(START_TOKEN)] + tokens
                tokens = [Token('[CLS]')] + tokens

                if self._max_len is not None:
                    tokens = tokens[:self._max_len]

                simplabel = None if simplabel is None else simplabel[0]

                words = [x.text for x in tokens]
                instance = self.text_to_instance(tokens, simplabel, words)
                if instance:
                    yield instance
        else:
            with open(file_path, "r", encoding='UTF-8') as data_file:
                logger.info("Reading instances from lines in file at: %s", file_path)
                for line in data_file:
                    line = line.strip("\n")
                    # skip blank and broken lines
                    if not line or (not self._test_mode and self._broken_dot_strategy == 'skip'
                                    and self.BROKEN_SENTENCES_REGEXP.search(line) is not None):
                        continue
                    sentence, simplabel = line.split('\t')
                    tokens_ = sentence.split(' ')
                    tokens = [Token(i) for i in tokens_]
                    simplabel = [simplabel]

                    # if tokens and tokens[0] != Token(START_TOKEN):
                    #     tokens = [Token(START_TOKEN)] + tokens
                    tokens = [Token('[CLS]')] + tokens

                    if self._max_len is not None:
                        tokens = tokens[:self._max_len]

                    simplabel = None if simplabel is None else simplabel[0]

                    words = [x.text for x in tokens]
                    instance = self.text_to_instance(tokens, simplabel, words)
                    if instance:
                        yield instance

    def text_to_instance(self, tokens: List[Token], simplabel: List[str] = None,
                         words: List[str] = None) -> Instance:  # type: ignore
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.
        """
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        sequence = TextField(tokens, self._token_indexers)
        fields["tokens"] = sequence
        fields["metadata"] = MetadataField({"words": words})
        fields["simplabels"] = LabelField(simplabel, label_namespace='simplabels')
        # print('fields["tokens"]', fields["tokens"])  # fields["tokens"] TextField of length 8 with text:  		[$START, Protests, across, the, nation, were, suppressed, .] 		and TokenIndexers : {'tokens': 'SingleIdTokenIndexer'}
        # print('fields["metadata"]', fields["metadata"])  # fields["metadata"] MetadataField (print field.metadata to see specific information).
        # print('fields["simplabel"]', fields["simplabel"])  # fields["simplabel"] LabelField with label: 0 in namespace: 'simplabel'.'

        return Instance(fields)



# For generator training only
#   input: sentences from source/target corpus
#   output: tokens, metadata, simplabels(from source/target corpus)
@DatasetReader.register("generator_datareader")
class GeneratorDatasetReader(DatasetReader):
    """
    Reads instances from a pretokenised file where each line is in the following format:

    WORD WORD WORD WORD WORD WORD WORD \n

    and converts it into a ``Dataset`` suitable for simplabel prediction. You can also specify
    alternative delimiters in the constructor.

    Parameters
    ----------
    delimiters: ``dict``
        The dcitionary with all delimeters.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
        Note that the `output` tags will always correspond to single token IDs based on how they
        are pre-tokenised in the data file.
    max_len: if set than will truncate long sentences
    """
    # fix broken sentences mostly in Lang8
    BROKEN_SENTENCES_REGEXP = re.compile(r'\.[a-zA-RT-Z]')

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 delimeters: dict = SEQ_DELIMETERS,
                 skip_correct: bool = False,
                 skip_complex: int = 0,
                 lazy: bool = False,
                 max_len: int = None,
                 test_mode: bool = False,
                 tag_strategy: str = "keep_one",
                 tn_prob: float = 0,
                 tp_prob: float = 0,
                 broken_dot_strategy: str = "keep") -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._delimeters = delimeters
        self._max_len = max_len
        self._skip_correct = skip_correct
        self._skip_complex = skip_complex
        self._tag_strategy = tag_strategy
        self._broken_dot_strategy = broken_dot_strategy
        self._test_mode = test_mode
        self._tn_prob = tn_prob
        self._tp_prob = tp_prob

    @overrides
    def _read(self, file_path):
        if 'src' in file_path or 'source' in file_path or 'difficult' in file_path:
            source_or_target = 'source'
        elif 'dst' in file_path or 'target' in file_path or 'simple' in file_path:
            source_or_target = 'target'
        else:
            print('Training or validating should provide source_or_target! ')
        file_path = cached_path(file_path)
        with open(file_path, "r", encoding='UTF-8') as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line in data_file:
                line = line.strip("\n")
                # skip blank and broken lines
                if not line or (not self._test_mode and self._broken_dot_strategy == 'skip'
                                and self.BROKEN_SENTENCES_REGEXP.search(line) is not None):
                    continue
                sentence = line
                tokens_ = sentence.split(' ')
                tokens = [Token(i) for i in tokens_]
                if source_or_target == 'source' or source_or_target is None:
                    simplabel = 'unsimple'
                else:
                    simplabel = 'simple'

                # if tokens and tokens[0] != Token(START_TOKEN):
                #     tokens = [Token(START_TOKEN)] + tokens
                tokens = [Token('[CLS]')] + tokens

                if self._max_len is not None:
                    tokens = tokens[:self._max_len]
                words = [x.text for x in tokens]
                instance = self.text_to_instance(tokens, simplabel, words)
                if instance:
                    yield instance

    def text_to_instance(self, tokens: List[Token], simplabel: List[str] = None,
                         words: List[str] = None) -> Instance:  # type: ignore
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.
        """
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        sequence = TextField(tokens, self._token_indexers)
        fields["tokens"] = sequence
        fields["metadata"] = MetadataField({"words": words})
        fields["simplabels"] = LabelField(simplabel, label_namespace='simplabels')
        # print('fields["tokens"]', fields["tokens"])  # fields["tokens"] TextField of length 8 with text:  		[$START, Protests, across, the, nation, were, suppressed, .] 		and TokenIndexers : {'tokens': 'SingleIdTokenIndexer'}
        # print('fields["metadata"]', fields["metadata"])  # fields["metadata"] MetadataField (print field.metadata to see specific information).
        # print('fields["simplabel"]', fields["simplabel"])  # fields["simplabel"] LabelField with label: 0 in namespace: 'simplabel'.'

        return Instance(fields)


@DatasetReader.register("generator_llm_datareader")
class GeneratorLLMDatasetReader(DatasetReader):
    # fix broken sentences mostly in Lang8
    BROKEN_SENTENCES_REGEXP = re.compile(r'\.[a-zA-RT-Z]')

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 delimeters: dict = SEQ_DELIMETERS,
                 skip_correct: bool = False,
                 skip_complex: int = 0,
                 lazy: bool = False,
                 max_len: int = None,
                 test_mode: bool = False,
                 tag_strategy: str = "keep_one",
                 tn_prob: float = 0,
                 tp_prob: float = 0,
                 broken_dot_strategy: str = "keep") -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._delimeters = delimeters
        self._max_len = max_len
        self._skip_correct = skip_correct
        self._skip_complex = skip_complex
        self._tag_strategy = tag_strategy
        self._broken_dot_strategy = broken_dot_strategy
        self._test_mode = test_mode
        self._tn_prob = tn_prob
        self._tp_prob = tp_prob

    def find_llm_cwi_form_token(self, text_token, cwi_list):
        if cwi_list == []:
            # print('kong')
            cwi_token = []
            marked_token = ['[DELETE]' if word in cwi_token else '[KEEP]' for word in text_token]
            return marked_token
        # print('text_token', text_token)
        cwi_words = [word for item in cwi_list for word in item.split()]
        # print('cwi_words', cwi_words)
        # cwi_list = ' '.join(cwi_list)
        # tokens = [Token(i) for i in cwi_list]
        # cwi_token = TextField(tokens, self._token_indexers)
        # print('cwi_token', cwi_token)
        # print('transmitted' in cwi_words)
        # marked_token = []
        # for word in text_token:
        #     print(word)
        #     if str(word) in cwi_words:
        #         print('1')
        #         marked_token.append('[DELETE]')
        #     else:
        #         print('2')
        #         marked_token.append('[KEEP]')
        marked_token = ['[DELETE]' if str(word) in cwi_words else '[KEEP]' for word in text_token]
        # print('marked_token', marked_token)
        return marked_token

    @overrides
    def _read(self, file_path):
        '''
        file_path下的文件结构：
        1. 简单的文本-txt格式 (simple)
            text\n
            ...
        2.复杂的文本-json格式，包含llm_cwi (unsimple)
            {
                "sentence": "Wikipedia is free content that anyone can edit and distribute .",
                "difficult_words": [
                    "content",
                    "distribute"
                ]
            },
            ...
        '''

        if 'gpt' in file_path or 'src' in file_path or 'source' in file_path or 'difficult' in file_path:
            source_or_target = 'source'
        elif 'dst' in file_path or 'target' in file_path or 'simple' in file_path:
            source_or_target = 'target'
        else:
            print('Training or validating should provide source_or_target! ')
        # 读取txt格式文件，该文件包括train、valid、test的所有target数据
        if file_path[-4:] == '.txt':
            # print('读取txt文件')
            file_path = cached_path(file_path)
            with open(file_path, "r", encoding='UTF-8') as data_file:
                logger.info("Reading instances from lines in file at: %s", file_path)
                # print("Reading instances from lines in file at:", file_path)
                for line in data_file:
                    line = line.strip("\n")
                    # skip blank and broken lines
                    if not line or (not self._test_mode and self._broken_dot_strategy == 'skip'
                                    and self.BROKEN_SENTENCES_REGEXP.search(line) is not None):
                        continue
                    sentence = line
                    tokens_ = sentence.split(' ')
                    tokens = [Token(i) for i in tokens_]
                    if source_or_target == 'source' or source_or_target is None:
                        simplabel = 'unsimple'
                    else:
                        simplabel = 'simple'

                    # if tokens and tokens[0] != Token(START_TOKEN):
                    #     tokens = [Token(START_TOKEN)] + tokens
                    tokens = [Token('[CLS]')] + tokens

                    if self._max_len is not None:
                        tokens = tokens[:self._max_len]
                    words = [x.text for x in tokens]
                    instance = self.text_to_instance(tokens, simplabel, words, cwi=None)
                    if instance:
                        yield instance
        # 读取json格式文件，该文件包括train、valid、test的所有source数据，带llm的cwi数据
        # 注意：格式为{"sentence"：xxx,
        #            "difficult_words": ['w', 'w', ...] or [] or null
        #            }
        # 其中，['w', 'w', ...]表示llm预测的cwi；
        # []表示llm认为该句子中无难词；
        # null表示llm未对该句子进行预测
        else:
            # print('读取json文件')
            with open(file_path, 'r', encoding='utf-8') as file:
                data_file = json.load(file)
                logger.info("Reading instances from lines in file at: %s", file_path)
                # print("Reading instances from lines in file at:", file_path)
                for line in data_file:
                    sentence = line["sentence"]
                    tokens_ = sentence.split(' ')
                    tokens = [Token(i) for i in tokens_]
                    # print('tokens', tokens)
                    cwi = line["difficult_words"]
                    # print('cwi', cwi)

                    if source_or_target == 'source' or source_or_target is None:
                        simplabel = 'unsimple'
                    else:
                        simplabel = 'simple'
                    # if tokens and tokens[0] != Token(START_TOKEN):
                    #     tokens = [Token(START_TOKEN)] + tokens
                    tokens = [Token('[CLS]')] + tokens

                    if self._max_len is not None:
                        tokens = tokens[:self._max_len]
                    words = [x.text for x in tokens]
                    instance = self.text_to_instance(tokens, simplabel, words, cwi=cwi)
                    if instance:
                        yield instance

    def text_to_instance(self, tokens: List[Token], simplabel: List[str] = None,
                         words: List[str] = None, cwi: List[str] = None) -> Instance:  # type: ignore
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.
        """
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        sequence = TextField(tokens, self._token_indexers)
        fields["tokens"] = sequence
        fields["metadata"] = MetadataField({"words": words})
        fields["simplabels"] = LabelField(simplabel, label_namespace='simplabels')
        if cwi == None:
            cwi = []
            keep_delete_label = self.find_llm_cwi_form_token(sequence, cwi)
            # print('keep_delete_label', keep_delete_label)
            # exit()
            fields["labels"] = SequenceLabelField(keep_delete_label, sequence, label_namespace="pipeline_tags")
            # fields["labels"] = None
        else:
            keep_delete_label = self.find_llm_cwi_form_token(sequence, cwi)
            # print('keep_delete_label', keep_delete_label)
            # exit()
            fields["labels"] = SequenceLabelField(keep_delete_label, sequence, label_namespace="pipeline_tags")
        # print('fields["labels"]', fields["labels"])
        # print('fields["tokens"]', fields["tokens"])  # fields["tokens"] TextField of length 8 with text:  		[$START, Protests, across, the, nation, were, suppressed, .] 		and TokenIndexers : {'tokens': 'SingleIdTokenIndexer'}
        # print('fields["metadata"]', fields["metadata"])  # fields["metadata"] MetadataField (print field.metadata to see specific information).
        # print('fields["simplabel"]', fields["simplabel"])  # fields["simplabel"] LabelField with label: 0 in namespace: 'simplabel'.'

        return Instance(fields)
