'''
data-LS测试数据：
    data-LS/BenchLS/BenchLS_929.json
    data-LS/LexMTurk/lex.mturk_500.json
    data-LS/NNSeval/NNSeval_239.json
    data-LS/TSAR-2022-en/tsar2022_en_test_gold_373.json
模型：
    gector.gec_model_without_append_adv_pipeline.GecBERTModel
        gector.seq2tag_pipeline_model.Seq2Tag_pineline

'''

import argparse
import json
from gector.gec_model_without_append_adv_pipeline import GecBERTModel

def complex_word_find(orig_sent, post_tag_batch, probability_all):
    complex_word_list = []
    cwi_probability_all = []
    # print('orig_sent', len(orig_sent))
    # print('post_tag_batch', len(post_tag_batch))
    # print('probability_all', len(probability_all), len(probability_all[0]))
    assert len(orig_sent) == len(post_tag_batch) == len(probability_all)
    for orig_tokens, post_tag_tokens, probability in zip(orig_sent, post_tag_batch, probability_all):
        orig_tokens = orig_tokens.split(' ')
        post_tag_tokens = post_tag_tokens.split(' ')
        complex_word_list.append([])
        assert len(orig_tokens) == len(post_tag_tokens)
        cwi_probability = []
        for i in range(len(orig_tokens)):
            if post_tag_tokens[i] == '[MASK]' or post_tag_tokens[i] == '[mask]':
                cwi_probability.append(probability[i+1][0])
                # 对被tokenizer拆分的词进行特殊处理
                if orig_tokens[i].startswith("##"):
                    if orig_tokens[i-1].startswith("##"):
                        complex_word_list[-1].append(orig_tokens[i - 2] + orig_tokens[i - 1].replace("##", "") + orig_tokens[i].replace("##", ""))
                    elif orig_tokens[i+1].startswith("##"):
                        complex_word_list[-1].append(orig_tokens[i - 1] + orig_tokens[i].replace("##", "") + orig_tokens[i+1].replace("##",  ""))
                    else:
                        complex_word_list[-1].append(orig_tokens[i-1] + orig_tokens[i].replace("##", ""))

                elif i+1 <= len(orig_tokens)-1 and orig_tokens[i+1].startswith("##"):
                    if i+2 <= len(orig_tokens)-1 and orig_tokens[i+2].startswith("##"):
                        complex_word_list[-1].append(orig_tokens[i] + orig_tokens[i+1].replace("##", "") + orig_tokens[i + 2].replace("##", ""))
                    else:
                        complex_word_list[-1].append(orig_tokens[i] + orig_tokens[i+1].replace("##", ""))
                else:
                    complex_word_list[-1].append(orig_tokens[i])
        cwi_probability_all.append(cwi_probability)
        assert len(cwi_probability) == len(complex_word_list[-1])

    assert len(complex_word_list) == len(orig_sent)
    complex_word_list_combine = []
    for i in range(len(complex_word_list)):
        complex_word_list_combine.append(' '.join(complex_word_list[i]))
    # print('cwi_probability_all', cwi_probability_all[0])
    return complex_word_list_combine, cwi_probability_all

def write_json_for(dic, filename):
    d_list = []
    # print(dic.keys())
    for a, b, s, t, p, c, e in zip(dic["unsimple_sentence "], dic["confusion_sentence"],
                                   dic["original_tokens "], dic["post_tag_tokens "],
                                   dic["keep_delete_tags"], dic["complex_words_list"], dic["probability_words"]):
        d = {
            "unsimple_sentence ": a,
            "confusion_sentence": b,
            "original_tokens   ": s,
            "post_tag_tokens   ": t,
            "keep_delete_tags  ": p,
            "complex_words_list": c,
            "probability_words": e
        }
        d_list.append(d)
    # print(d_list)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(d_list, f, indent=4, ensure_ascii=False)
    f.close()

def read_lines(input_file):
    # json数据读取
    if input_file[-4:] == 'json':
        test_data = []
        with open(input_file, 'r', encoding='utf-8') as fp:
            data_dict = json.load(fp)
            for dic in data_dict:
                test = dic["unsimple_sentence"]
                # print('test', test)
                test_data.append(test)
        # print('test_data', test_data, len(test_data))
    # txt数据读取
    else:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        test_data = [line.strip() for line in lines]
    print('测试数据数量：', str(len(test_data)))
    return test_data

def without_pad(input_list, clean=False):
    result_lines = []
    for p in input_list:
        temp = []
        for pp in p[1:]:
            if pp != '[PAD]':
                temp.append(pp)
            else:
                break
        if clean:
            out_string = " ".join(temp).replace(" ##", "").strip()
        else:
            out_string = " ".join(temp).strip()
        result_lines.append(out_string)
    return result_lines

def predict_for_file(input_file, output_file, model, batch_size=32):
    test_data = read_lines(input_file)
    predictions = []
    tags_all = []
    cnt_corrections = 0
    batch = []
    orig_batch_all = []
    probability_all = []
    for sent in test_data:
        batch.append(sent.split())
        if len(batch) == batch_size:
            print('Handling batch...')
            orig_batch, post_tag_batch, tags, logits_labels, class_probabilities_labels = model.handle_batch(batch, need_probability=True)
            # print('class_probabilities_labels', class_probabilities_labels)
            predictions.extend(post_tag_batch)
            orig_batch_all.extend(orig_batch)
            # print('tags', tags)
            tags_all.extend(tags)
            probability_all.extend(class_probabilities_labels)
            # print(len(class_probabilities_labels), len(class_probabilities_labels[0]))
            batch = []
            # exit()
    if batch:
        print('Handling batch...')
        orig_batch, post_tag_batch, tags, logits_labels, class_probabilities_labels = model.handle_batch(batch, need_probability=True)
        # print('class_probabilities_labels', class_probabilities_labels)
        predictions.extend(post_tag_batch)
        orig_batch_all.extend(orig_batch)
        tags_all.extend(tags)
        # print(len(class_probabilities_labels), len(class_probabilities_labels[0]))
        probability_all.extend(class_probabilities_labels)

    # print('predictions', predictions)
    # print('tags_all', tags_all)
    post_tag_batch_ = without_pad(predictions, clean=True)
    post_tag_batch = without_pad(predictions, clean=False)
    orig_token = []
    for orig_batch in orig_batch_all:
        # print('orig_batch', orig_batch)
        temp = []
        for j in orig_batch:
            temp.append(model.vocab.get_token_from_index(int(j), namespace='labels'))
        orig_token.append(temp)
    orig_sent = without_pad(orig_token, clean=False)
    orig_sent_ = without_pad(orig_token, clean=True)
    tags_all = without_pad(tags_all)

    # 进行复杂词提取步骤
    print('orig_sent', len(orig_sent))
    print('post_tag_batch', len(post_tag_batch))
    print('probability_all', len(probability_all), len(probability_all[0]))
    complex_words_list, cwi_probability_all = complex_word_find(orig_sent, post_tag_batch, probability_all)

    dic = {
        "unsimple_sentence ": orig_sent_,
        "confusion_sentence": post_tag_batch_,
        "original_tokens ": orig_sent,
        "post_tag_tokens ": post_tag_batch,
        "keep_delete_tags": tags_all,
        "complex_words_list": complex_words_list,
        "probability_words": cwi_probability_all,
    }

    write_json_for(dic, output_file)
    return dic



def main(args):
    # get all paths
    model = GecBERTModel(vocab_path=args.vocab_path,
                         model_paths=[args.model_path],
                         max_len=args.max_len, min_len=args.min_len,
                         iterations=args.iteration_count,
                         lowercase_tokens=args.lowercase_tokens,
                         model_name=args.transformer_model,
                         special_tokens_fix=args.special_tokens_fix,
                         log=False,
                         is_ensemble=args.is_ensemble,
                         weigths=args.weights,
                         pre_pretrained_model_path=args.pre_pretrained_model_path)

    dic_out = predict_for_file(args.input_file, args.output_file, model,
                                       batch_size=args.batch_size)
    print('Finished!')
    # evaluate with m2 or ERRANT
    # print(f"Produced overall corrections: {cnt_corrections}")


if __name__ == '__main__':
    # read parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        help='Path to the model file.',
                        default='finished_trained_models/model_6_14/generator_14.th')
    parser.add_argument('--vocab_path',
                        help='Path to the model file.',
                        default='saved_dict_adv/vocabulary'  # to use pretrained models
                        )
    parser.add_argument('--input_file',
                        help='Path to the eval set file',
                        default='data-LS/BenchLS/BenchLS_929.json')
    parser.add_argument('--output_file',
                        help='Path to the output file',
                        default='data-LS/BenchLS/predict.json')
    parser.add_argument('--max_len',
                        type=int,
                        help='The max sentence length'
                             '(all longer will be truncated)',
                        default=500)
    parser.add_argument('--min_len',
                        type=int,
                        help='The minimum sentence length'
                             '(all longer will be returned w/o changes)',
                        default=1)
    parser.add_argument('--batch_size',
                        type=int,
                        help='The size of hidden unit cell.',
                        default=32)
    parser.add_argument('--lowercase_tokens',
                        type=int,
                        help='Whether to lowercase tokens.',
                        default=1)
    parser.add_argument('--transformer_model',
                        choices=['bert', 'gpt2', 'transformerxl', 'xlnet', 'distilbert', 'roberta', 'albert'
                                 'bert-large', 'roberta-large', 'xlnet-large'],
                        help='Name of the transformer model.',
                        default='bert')
    parser.add_argument('--iteration_count',
                        type=int,
                        help='The number of iterations of the model.',
                        default=1)
    parser.add_argument('--special_tokens_fix',
                        type=int,
                        help='Whether to fix problem with [CLS], [SEP] tokens tokenization. '
                             'For reproducing reported results it should be 0 for BERT/XLNet and 1 for RoBERTa.',
                        default=0)
    parser.add_argument('--is_ensemble',
                        type=int,
                        help='Whether to do ensembling.',
                        default=0)
    parser.add_argument('--weights',
                        help='Used to calculate weighted average', nargs='+',
                        default=None)
    parser.add_argument('--pre_pretrained_model_path',
                        help='Path to the pretrained model file related to --transformer_model. (None for auto downloading from huggingface web.)',
                        default='pre_pretrained_models/bert')
    parser.add_argument('--delete_threshold',
                        type=float,
                        help='The threshold for model to [delete]([mask]).',
                        default=0.0)
    args = parser.parse_args()
    main(args)
