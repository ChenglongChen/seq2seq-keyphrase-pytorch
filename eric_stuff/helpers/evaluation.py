import time
import torch
import math
import numpy as np
from helpers.helper import generator_queue
wait_time = 0.01  # in seconds


def eval_teacher_forcing(_model, batch_generator, number_batch, criterion):
    _model.eval()
    data_queue, _ = generator_queue(batch_generator, max_q_size=20)
    sum_loss = 0.0
    for i in range(number_batch):
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
