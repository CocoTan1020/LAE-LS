import argparse

from sklearn import metrics
import json
import sys
import re
import os
from nltk.corpus import names


def lower_list(word_list):
    if type(word_list[0]) == type([]):
        word_list_new = []
        for wl in word_list:
            word_list_new.append([i.lower() for i in wl])
        return word_list_new
    else:
        return [i.lower() for i in word_list]


def report_score_sg(unsimple_word_list, truth_substitute_list, predicted_substitute_list, file_path=None):
    if file_path:
        sys.stdout = open(file_path, 'a', encoding='utf-8')
    assert len(unsimple_word_list) == len(truth_substitute_list) == len(predicted_substitute_list)
    potential = 0
    instances = len(unsimple_word_list)
    precision = 0
    precision_all = 0
    recall = 0
    recall_all = 0
    for i in range(len(unsimple_word_list)):
        common = list(set(predicted_substitute_list[i]).intersection(truth_substitute_list[i]))
        if len(common) >= 1:
            potential += 1
        precision += len(common)
        recall += len(common)
        precision_all += 10 if len(predicted_substitute_list[i]) != 0 else 0
        recall_all += len(truth_substitute_list[i])
    potential /= instances
    precision /= precision_all
    recall /= recall_all
    if precision + recall == 0:
        F_score = 0
    else:
        F_score = 2 * precision * recall / (precision + recall)
    print("{:^10}{:^10}{:^10}{:^10}".format("Potential", "Precision", "Recall", "F1"))
    print('-' * 50)
    print("{:^10.4f}\t{:^10.4f}\t{:^10.4f}\t{:^10.4f}".format(potential, precision, recall, F_score))
    print()
    print()
    if file_path:
        sys.stdout.close()
        sys.stdout = sys.__stdout__
    return potential, precision, recall, F_score


def string_list(prediced_simple_word):
    prediced_simple_word = prediced_simple_word.strip("[]")
    prediced_simple_word = prediced_simple_word.split(", ")
    return prediced_simple_word


def predict_label(predict_json, truth_json):
    unsimple_word_list = []
    truth_substitute_list = []
    predicted_substitute_list = []
    with open(truth_json, 'r', encoding='utf-8') as file1:
        with open(predict_json, 'r', encoding='utf-8') as file2:
            data1 = json.load(file1)
            data2 = json.load(file2)
            for obj1, obj2 in zip(data1, data2):
                unsimple_word = obj1["unsimple_word"]  # "unsimple_word": "prominent"
                simple_word_truth = obj1['simple_word']  # "simple_word": ["important", "noticeable",...]
                if unsimple_word in obj2['difficult_words']:
                    try:
                        prediced_simple_word = obj2['simplified_words'][unsimple_word]
                        prediced_simple_word = string_list(prediced_simple_word)[:10]
                    except KeyError:
                        print('error')
                        prediced_simple_word = []
                else:
                    prediced_simple_word = []
                unsimple_word_list.append(unsimple_word)
                truth_substitute_list.append(simple_word_truth)
                predicted_substitute_list.append(prediced_simple_word)
    return unsimple_word_list, truth_substitute_list, predicted_substitute_list


def predict_label2(path, predict_json, truth_json):
    unsimple_word_list = []
    truth_substitute_list = []
    predicted_substitute_list = []
    with open(path, 'r', encoding='utf-8') as file:
        with open(truth_json, 'r', encoding='utf-8') as file1:
            with open(predict_json, 'r', encoding='utf-8') as file2:
                data = json.load(file)
                data1 = json.load(file1)
                data2 = json.load(file2)
                for obj, obj1, obj2 in zip(data, data1, data2):
                    unsimple_word = obj1["unsimple_word"]  # "unsimple_word": "prominent"
                    simple_word_truth = obj1['simple_word']  # "simple_word": ["important", "noticeable",...]
                    if unsimple_word in obj['unsimple_word_predict']:
                        try:
                            prediced_simple_word = obj2['simplified_words'][unsimple_word]
                            prediced_simple_word = string_list(prediced_simple_word)[:10]
                        except KeyError:
                            prediced_simple_word = []
                    else:
                        prediced_simple_word = []
                    unsimple_word_list.append(unsimple_word)
                    truth_substitute_list.append(simple_word_truth)
                    predicted_substitute_list.append(prediced_simple_word)
    return unsimple_word_list, truth_substitute_list, predicted_substitute_list


def report_score_cwi(unsimple_word_list_true, unsimple_word_list_predict, Fscore='F1'):
    precision = 0
    precision_all = 0
    recall = 0
    recall_all = 0
    for true, pred in zip(unsimple_word_list_true, unsimple_word_list_predict):
        if true in pred:
            precision += 1
            recall += 1
        precision_all += len(set(pred))
        recall_all += 1
    if precision_all == 0:
        precision = 0
    else:
        precision /= precision_all
    if recall_all == 0:
        recall = 0
    else:
        recall /= recall_all
    if precision == 0 and recall == 0:
        F_score = 0
    else:
        if Fscore == 'F1':
            F_score = 2 * precision * recall / (precision + recall)
        elif Fscore == 'F2':
            F_score = (1 + 2 ** 2) * precision * recall / (2 ** 2 * precision + recall)
    print("{:^10}{:^10}{:^10}".format("Precision", "Recall", Fscore))
    print('-' * 50)
    print("{:^10.4f}\t{:^10.4f}\t{:^10.4f}".format(precision, recall, F_score))
    print()
    print()


def find_unsimple3(predict_json, truth_json):
    unsimple_word_list_true = []
    unsimple_word_list_predict = []
    with open(truth_json, 'r', encoding='utf-8') as file1:
        with open(predict_json, 'r', encoding='utf-8') as file2:
            data1 = json.load(file1)
            data2 = json.load(file2)
            for obj1, obj2 in zip(data1[:246], data2[:246]):
                unsimple_word = obj1["unsimple_word"]
                unsimple_word_pre = obj2["difficult_words"]  # unsimple_word_predict, difficult_words

                unsimple_word_list_true.append(unsimple_word)
                unsimple_word_list_predict.append(unsimple_word_pre)
    return unsimple_word_list_true, unsimple_word_list_predict


def read_json_to_list(json_path):
    dic_all = []
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        for obj in data:
            dic_all.append(obj)
    return dic_all


def predict_label_combine(truth_json1, predict_json1, truth_json2, predict_json2):
    unsimple_word_list = []
    truth_substitute_list = []
    predicted_substitute_list = []
    truth_1 = read_json_to_list(truth_json1)
    predict_1 = read_json_to_list(predict_json1)
    truth_2 = read_json_to_list(truth_json2)
    predict_2 = read_json_to_list(predict_json2)
    for truth_1_, predict_1_ in zip(truth_1, predict_1):
        if truth_1_['unsimple_word'] in predict_1_['difficult_words']:
            unsimple_word_list.append(truth_1_['unsimple_word'])
            truth_substitute_list.append(truth_1_['simple_word'])
            prediced_simple_word = predict_1_['simplified_words'][truth_1_['unsimple_word']]
            prediced_simple_word = string_list(prediced_simple_word)[:10]
            predicted_substitute_list.append(prediced_simple_word)
        else:
            continue
    for truth_2_, predict_2_ in zip(truth_2, predict_2):
        unsimple_word_list.append(truth_2_['unsimple_word'])
        truth_substitute_list.append(truth_2_['simple_word'])
        prediced_simple_word = predict_2_['simplified_words']
        prediced_simple_word = prediced_simple_word[:10]
        predicted_substitute_list.append(prediced_simple_word)
    return unsimple_word_list, truth_substitute_list, predicted_substitute_list


def true_ourmodel_cwi(path, predict_json, truth_json):
    unsimple_word_list = []
    truth_substitute_list = []
    predicted_substitute_list = []
    with open(path, 'r', encoding='utf-8') as file:
        with open(predict_json, 'r', encoding='utf-8') as file1:
            with open(truth_json, 'r', encoding='utf-8') as file2:
                data = json.load(file)
                data1 = json.load(file1)
                data2 = json.load(file2)
                for obj, obj1, obj2 in zip(data, data1, data2):
                    unsimple_word = obj1["unsimple_word"]  # "unsimple_word": "prominent"
                    simple_word_truth = obj1['simple_word']  # "simple_word": ["important", "noticeable",...]
                    if unsimple_word in obj['unsimple_word_predict']:  # unsimple_word_predict, difficult_words
                        try:
                            prediced_simple_word = obj2['simplified_words'][unsimple_word]
                            prediced_simple_word = string_list(prediced_simple_word)[:10]
                        except KeyError:
                            prediced_simple_word = []
                    else:
                        prediced_simple_word = []
                    # print(prediced_simple_word)
                    unsimple_word_list.append(unsimple_word)
                    truth_substitute_list.append(simple_word_truth)
                    predicted_substitute_list.append(prediced_simple_word)
    return unsimple_word_list, truth_substitute_list, predicted_substitute_list


def LS_evaluation_before(truth_json, predict_json):
    TP, FP, TN, FN = 0, 0, 0, 0
    with open(truth_json, 'r', encoding='utf-8') as file1:
        with open(predict_json, 'r', encoding='utf-8') as file2:
            data1 = json.load(file1)
            data2 = json.load(file2)
            for obj1, obj2 in zip(data1, data2):
                sentence = obj1['unsimple_sentence']
                true_unsimple_word = obj1['unsimple_word']
                true_simple_word = obj1['simple_word']
                predict_difficult_words = obj2['difficult_words']
                predict_simplified_words = obj2['simplified_words']
                if true_unsimple_word in predict_difficult_words:
                    pred_simple = string_list(predict_simplified_words[true_unsimple_word])[0]
                    if pred_simple in true_simple_word:  # 找到难词且改对
                        TP += 1
                    else:  # 找到难词且改错
                        FP += 1
                    TN += len(sentence.split(' ')) - len(predict_difficult_words) + 1  # 简单词不改
                    FN += len(predict_difficult_words) - 1  # 简单词改了
                else:
                    FP += 1  # 没找到难词
                    TN += len(sentence.split(' ')) - len(predict_difficult_words)  # 简单词不改
                    FN += len(predict_difficult_words)  # 简单词改了
    print('TP: ', TP)
    print('FP: ', FP)
    print('TN: ', TN)
    print('FN: ', FN)
    precision = TP / (TP + FN)
    recall = TP / (TP + FP)
    F1 = 2 * precision * recall / (precision + recall)
    print("{:^10}{:^10}{:^10}".format("Precision", "Recall", "F1"))
    print('-' * 50)
    print("{:^10.4f}\t{:^10.4f}\t{:^10.4f}".format(precision, recall, F1))
    print()
    return precision, recall, F1


def LS_evaluation(truth_json, predict_json):
    precision, precision_all, recall, recall_all = 0, 0, 0, 0
    with open(truth_json, 'r', encoding='utf-8') as file1:
        with open(predict_json, 'r', encoding='utf-8') as file2:
            data1 = json.load(file1)
            data2 = json.load(file2)
            for obj1, obj2 in zip(data1, data2):
                sentence = obj1['unsimple_sentence']
                true_unsimple_word = obj1['unsimple_word']
                true_simple_word = obj1['simple_word']
                predict_difficult_words = obj2['difficult_words']
                predict_simplified_words = obj2['simplified_words']
                if true_unsimple_word in predict_difficult_words:
                    # print(true_unsimple_word)
                    # print(predict_difficult_words)
                    pred_simple = string_list(predict_simplified_words[true_unsimple_word])[0]
                    # print(pred_simple)
                    # print(true_simple_word)
                    if pred_simple in true_simple_word:  # 找到难词且改对
                        precision += 1
                        recall += 1
                precision_all += len(predict_difficult_words)
                recall_all += 1
    print('precision: ', precision)
    print('precision_all: ', precision_all)
    print('recall: ', recall)
    print('recall_all: ', recall_all)
    precision = precision / precision_all
    recall = recall / recall_all
    F1 = 2 * precision * recall / (precision + recall)
    print("{:^10}{:^10}{:^10}".format("Precision", "Recall", "F1"))
    print('-' * 50)
    print("{:^10.4f}\t{:^10.4f}\t{:^10.4f}".format(precision, recall, F1))
    print()
    return precision, recall, F1


def LS_evaluation_(truth_json, predict_json, path):
    precision, precision_all, recall, recall_all = 0, 0, 0, 0
    with open(truth_json, 'r', encoding='utf-8') as file1:
        with open(predict_json, 'r', encoding='utf-8') as file2:
            with open(path, 'r', encoding='utf-8') as file3:
                data1 = json.load(file1)
                data2 = json.load(file2)
                data3 = json.load(file3)
                for obj1, obj2, obj3 in zip(data1, data2, data3):
                    true_unsimple_word = obj1['unsimple_word']
                    true_simple_word = obj1['simple_word']
                    predict_difficult_words = obj2['unsimple_word_predict']
                    predict_simplified_words = obj3['simplified_words']
                    if true_unsimple_word in predict_difficult_words:
                        pred_simple = string_list(predict_simplified_words[true_unsimple_word])[0]
                        # print(pred_simple)
                        # print(true_simple_word)
                        if pred_simple in true_simple_word:  # 找到难词且改对
                            precision += 1
                            recall += 1
                    precision_all += len(predict_difficult_words)
                    recall_all += 1
    print('precision: ', precision)
    print('precision_all: ', precision_all)
    print('recall: ', recall)
    print('recall_all: ', recall_all)
    precision = precision / precision_all
    recall = recall / recall_all
    F1 = 2 * precision * recall / (precision + recall)
    print("{:^10}{:^10}{:^10}".format("Precision", "Recall", "F1"))
    print('-' * 50)
    print("{:^10.4f}\t{:^10.4f}\t{:^10.4f}".format(precision, recall, F1))
    print()
    return precision, recall, F1


if __name__ == '__main__':
    pass
