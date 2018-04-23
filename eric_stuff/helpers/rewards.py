# coding=utf-8
import numpy as np
from collections import Counter
import itertools
from helpers.bleu.bleu import Bleu


class RewardCalculationRuntime(object):
    def __init__(self, word2id=None, id2word=None):
        self.word2id = word2id
        self.id2word = id2word

    def get_reward(self):
        raise NotImplementedError


class DuplicationReward(RewardCalculationRuntime):
    # find duplicate key phrases, assign minus rewards for duplications
    # returned vector should be same shape as input

    def get_reward(self, generated_list):
        # lol
        if len(generated_list) == 0:
            return [0.0]
        if len(generated_list) == 1:
            return [1.0]
        cache = set()
        res = []
        for g in generated_list:
            g = "___".join([str(word_id) for word_id in g])
            if g in cache:
                res.append(-1.0)
            else:
                res.append(1.0)
                cache.add(g)
        return res


class HardF1Reward(RewardCalculationRuntime):
    # match generated actions and ground truth on action level
    # instead of exact match, here we use max of f1 score between each other
    # returned vector should be same shape as input (list of token indices)
    def f1_score(self, prediction, ground_truth):
        if len(prediction) == 0 or len(ground_truth) == 0:
            return 0.0
        common = Counter(prediction) & Counter(ground_truth)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction)
        recall = 1.0 * num_same / len(ground_truth)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def get_reward(self, generated_list, ground_truth_list):
        # lol
        if len(generated_list) == 0 or len(ground_truth_list) == 0:
            return 0.0
        local_id2str = []
        local_str2id = {}
        generated_local_id_list = []
        for g in generated_list:
            g = "___".join([str(word_id) for word_id in g])
            if g not in local_str2id:
                local_str2id[g] = len(local_id2str)
                local_id2str.append(g)
            generated_local_id_list.append(local_str2id[g])
        ground_truth_local_id_list = []
        for g in ground_truth_list:
            if np.sum(g) == 0:
                continue
            g = np.trim_zeros(g, 'b')
            g = "___".join([str(word_id) for word_id in g])
            if g not in local_str2id:
                local_str2id[g] = len(local_id2str)
                local_id2str.append(g)
            ground_truth_local_id_list.append(local_str2id[g])

        score = self.f1_score(generated_local_id_list, ground_truth_local_id_list)
        return score


# class MultiSoftMatchReward(SoftMatchReward):
#     # match generated actions and ground truth on action level
#     # instead of exact match, here we use max of f1 score between each other
#     # returned vector should be same shape as input (list of token indices)
#     def get_reward(self, generated_actions, ground_truth_actions):
#         # lol
#         if len(generated_actions) == 1 and generated_actions[0] <= 4:
#             return [-1.0]
#         # add "--|--" before head, and remove "--EOS--" at tail
#         remove_last_token = True if generated_actions[-1] == 2 else False
#         if remove_last_token:
#             generated_actions = generated_actions[:-1]
#         score = self.max_f1_score(generated_actions, ground_truth_actions)
#         res = [score for _ in generated_actions]
#         if remove_last_token:
#             res += [0.5]
#         return res


# class MultiHardMatchReward(RewardCalculationRuntime):
#     # match generated actions and ground truth on action level
#     # returned vector should be same shape as input (list of token indices)
#     def get_reward(self, generated_actions, ground_truth_actions):
#         # lol
#         if len(generated_actions) == 1 and generated_actions[0] <= 4:
#             return 0.0
#         remove_last_token = True if generated_actions[-1] == 2 else False
#         if remove_last_token:
#             generated_actions = generated_actions[:-1]
#         # remove paddings from ground truth
#         gt = np.trim_zeros(ground_truth_actions, 'b').tolist()
#         gt = [4] + gt[:-1]
#         # hash ground truth
#         gt = self.group(gt, 4)
#         gt_map = set()
#         for item in gt:
#             if len(item) > 0:
#                 gt_map.add(str(item))
#         # check same
#         if str(generated_actions) in gt_map:
#             score = 1.0
#         else:
#             score = 0.0
#         res = [score for _ in generated_actions]
#         if remove_last_token:
#             res += [0.5]
#         return res


# class MultiBleuReward(RewardCalculationRuntime):
#     # hyp: string
#     # ref: list of strings
#     def bleu_score(self, hyp, ref):
#         ref = [a.strip() for a in ref]
#         refs = {0: ref}
#         hyps = {0: [hyp.strip()]}
#         scorer = Bleu(4)
#         score, _ = scorer.compute_score(refs, hyps)
#         return score  # list of 4 scores

#     def skip_3(self, sent):
#         word_list = sent.split()
#         if len(word_list) < 4:
#             return sent
#         return " ".join(word_list[:2] + word_list[-1:])

#     def get_reward(self, pred_string, gt_strings, ngram=1):
#         pred_string = self.skip_3(pred_string)
#         gt_strings = [self.skip_3(gt) for gt in gt_strings]
#         pred_length = min(len(pred_string.split()), 3)
#         score = self.bleu_score(pred_string, gt_strings)[ngram - 1]
#         res = [score for _ in range(pred_length)]
#         return res
