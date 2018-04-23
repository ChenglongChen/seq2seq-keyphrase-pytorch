import time
import torch
import math
from tqdm import tqdm
import numpy as np
from helpers.helper import generator_queue, pad_sequences, add_char_level_inputs
from helpers.rewards import HardF1Reward
from helpers.sampler import sample_a_batch
wait_time = 0.01  # in seconds


def eval_teacher_forcing(_model, batch_generator, number_batch, criterion):
    _model.eval()
    data_queue, _ = generator_queue(batch_generator, max_q_size=20)
    sum_loss = 0.0
    for i in tqdm(range(number_batch)):
        generator_output = None
        while True:
            if not data_queue.empty():
                generator_output = data_queue.get()
                break
            else:
                time.sleep(wait_time)
        input_source, input_target, input_prev_target, input_source_char, input_target_char, input_prev_target_char,\
            output_target, output_target_mask, local_dict = generator_output
        batch_size = input_source.size(0)

        history_info = _model.get_history_info(input_prev_target, input_prev_target_char)
        p_positions_mapped, p_target_vocab, _, _ = _model.forward(input_source, input_target, input_source_char, input_target_char, history_info)

        preds = p_target_vocab
        for p in p_positions_mapped:
            preds = preds + p
        preds = preds * output_target_mask.float().unsqueeze(-1)  # batch x time x vocab_size
        loss = criterion(preds, output_target, output_target_mask)  # batch
        loss = torch.mean(loss)  # 1

        batch_loss = loss.cpu().data.numpy()
        sum_loss += batch_loss * batch_size

    avg_loss = sum_loss / float(batch_size * (i + 1))
    avg_ppl = math.exp(avg_loss) if avg_loss < 13. else np.inf
    return avg_loss, avg_ppl


def eval_free_running(_model, data, batch_size, id2word, char2id, enable_cuda=False):
    _model.eval()
    data_size = len(data["input_source"])
    number_batch = (data_size + batch_size - 1) // batch_size
    f_score = []
    f_func = HardF1Reward()

    for i in tqdm(range(number_batch)):
        input_source = data["input_source"][i * batch_size: (i + 1) * batch_size]
        output_target = data["output_target"][i * batch_size: (i + 1) * batch_size]
        local_dict = data["local_oov_dict"][i * batch_size: (i + 1) * batch_size]

        input_source = pad_sequences(input_source, padding='post').astype('int32')  # batch x source_len
        input_source_char = add_char_level_inputs(input_source, id2word, char2id, local_dict)  # batch x source_len x char_len

        pred_word_ids, _ = sample_a_batch(_model, input_source, input_source_char, local_dict, id2word, char2id,
                                          sample=True, generate_this_many_peyphrases=8, max_keyphrase_length=5, enable_cuda=enable_cuda)
        # pred_word_ids: batch x n_keyphrase x n_word
        # output_target: batch x n_keyphrase x n_word
        for pred, gt in zip(pred_word_ids, output_target):
            f1 = f_func.get_reward(pred, gt)
            f_score.append(f1)
    avg_f1 = np.mean(f_score)
    return avg_f1
