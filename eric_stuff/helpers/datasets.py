# coding=utf-8
import random
import numpy as np
import torch
from helpers.helper import pad_sequences
np.random.seed(42)
random.seed(42)


def build_char_vocab(word_list):
    char2id = {}
    id2char = []
    for word in word_list:
        chars = list(word)
        for ch in chars:
            if ch not in char2id:
                char2id[ch] = len(id2char)
                id2char.append(ch)
    return char2id, id2char


def list_to_nparray(data):
    # only supports 2d lists
    assert isinstance(data, list) and len(data) > 0
    assert isinstance(data[0], list) and len(data[0]) > 0
    # 2d list, e.g., batch x source_len
    res = pad_sequences(data, padding='post').astype('int32')
    return res


def swap_0_1(thing):
    assert -1 not in thing
    thing[thing == 0] = -1
    thing[thing == 1] = 0
    thing[thing == -1] = 1
    return thing


def list_of_dict_to_dict_of_nparray(data_list):
    # this funcion also swaps 0 and 1s

    source, prev_target, target, oov_dict = [], [], [], []
    for item in data_list:
        src = np.array(item["src_oov"])
        src = swap_0_1(src).tolist()
        tgt = [np.array(t) for t in item["trg_copy"]]
        tgt = [swap_0_1(t).tolist() for t in tgt]

        sample_this_many_times = min(10, len(tgt))
        for i in range(sample_this_many_times):
            perm = np.random.permutation(len(tgt))
            _tgt = tgt[perm[0]]
            perm = perm[1:]
            if len(perm) == 0:
                prev = [0]
            else:
                how_many_prev = np.random.randint(len(perm))
                if how_many_prev == 0:
                    prev = [0]
                else:
                    prev = []
                    for j in range(how_many_prev):
                        prev += tgt[perm[j]]

            source.append(src)
            target.append(_tgt)
            prev_target.append(prev)
            oov_dict.append(item["oov_dict"])

    index_shuf = list(range(len(source)))
    random.shuffle(index_shuf)
    _source_shuf = [source[i] for i in index_shuf]
    _target_shuf = [target[i] for i in index_shuf]
    _prev_target_shuf = [prev_target[i] for i in index_shuf]
    _oov_dict_shuf = [oov_dict[i] for i in index_shuf]

    _source_shuf = list_to_nparray(_source_shuf)
    _target_shuf = list_to_nparray(_target_shuf)
    _prev_target_shuf = list_to_nparray(_prev_target_shuf)
    res = {'input_source': _source_shuf,
           'input_target': _target_shuf,
           'input_prev_target': _prev_target_shuf,
           'local_oov_dict': _oov_dict_shuf}
    return res


def load_dataset(path):
    # read vocab, train/valid/test set
    word2id, id2word, _ = torch.load(path + ".vocab.pt", 'wb')
    train = torch.load(path + ".valid.one2many.pt", 'wb')
    valid = torch.load(path + ".valid.one2many.pt", 'wb')
    test = torch.load(path + ".test.one2many.pt", 'wb')
    # use list instead of dictionary for id2word
    new_id2word = []
    for i in range(len(id2word)):
        new_id2word.append(id2word[i])
    char2id, id2char = build_char_vocab(new_id2word)
    # swap 0 and 1 in vocab and data
    # after doing this, 0 will be padding, and 1 will be start of sentence
    # 0: <pad>
    # 1: <s>
    # 2: </s>
    # 3: <unk>
    bos_token = new_id2word[0]
    pad_token = new_id2word[1]
    new_id2word[0] = pad_token
    new_id2word[1] = bos_token
    word2id[bos_token] = 1
    word2id[pad_token] = 0

    train_data = list_of_dict_to_dict_of_nparray(train)
    valid_data = list_of_dict_to_dict_of_nparray(valid)
    test_data = list_of_dict_to_dict_of_nparray(test)
    return word2id, new_id2word, char2id, id2char, train_data, valid_data, test_data
