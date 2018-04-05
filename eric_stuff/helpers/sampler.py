# coding=utf-8
import torch
import numpy as np
from helpers.helper import pad_sequences, to_pt, max_len, add_char_level_inputs


def sample_or_argmax(inputs, _model,
                     pad_id=0, bos_id=1, eos_id=2, unk_id=3,
                     id2word=None, char2id=None, max_sequence_length=5,
                     sample=False, enable_cuda=False):
    assert max_sequence_length > 0
    chosen, chosen_probs = [], []
    source, history, source_char, history_char, local_dict = inputs
    local_id2word = [{v: k for k, v in d.items()} for d in local_dict]
    batch_size = source.size(0)
    # use get_history_info() to get encode previously generated stuff
    history_info = _model.get_history_info(history, history_char)
    # use f_init() to get source encoding and init hidden state / cell state for decoder
    source_encoding_sequence, source_mask, next_h, next_c = _model.f_init(source, source_char)
    # begining of the sentence
    next_w = np.ones((batch_size, 1), dtype='int32') * bos_id  # batch x 1
    _temp_char_id_list = [char2id[ch] for ch in list('<s>')]
    next_w_char = np.array([_temp_char_id_list] * batch_size).astype('int32')
    next_w_char = np.reshape(next_w_char, (batch_size, 1, next_w_char.shape[-1]))  # batch x 1 x char_len

    for t in range(max_sequence_length):
        next_w = to_pt(next_w, enable_cuda=enable_cuda)
        next_w_char = to_pt(next_w_char, enable_cuda=enable_cuda)

        p_positions_mapped, p_target_vocab, next_h, next_c = _model.f_next(next_w, next_w_char,
                                                                           source, source_encoding_sequence, source_mask,
                                                                           history_info, next_h, next_c)

        pred = p_target_vocab
        for p in p_positions_mapped:
            pred = pred + p
        pred_cpu = pred.cpu().data.numpy()  # batch size x vocab_size

        if sample:
            choose_this_one = []
            for p in pred_cpu:
                if p.sum() == 0.0:
                    p[eos_id] = 1.0
                else:
                    p /= p.sum()
                choose_this_one.append(np.random.choice(len(p), 1, replace=False, p=p)[0])  # choice returns a list with one element in it
            choose_this_one = np.array(choose_this_one)  # batch
        else:
            choose_this_one = np.argmax(pred_cpu, axis=1)  # batch
        probs = torch.gather(pred, 1, to_pt(choose_this_one, enable_cuda=enable_cuda).contiguous().view(-1, 1))  # batch x 1
        probs = probs.squeeze()  # batch
        next_w = np.reshape(choose_this_one, choose_this_one.shape + (1,))

        sampled_words = []
        for i, w in enumerate(choose_this_one):
            sampled_words.append(id2word[w] if w < len(id2word) else local_id2word[i][w])  # batch of words
        next_w_char = [list(w.lower()) for w in sampled_words]  # list of list of chars
        next_w_char = [[char2id[ch] for ch in word] for word in next_w_char]  # list of list of char_ids
        next_w_char = pad_sequences(next_w_char, maxlen=max_len(next_w_char), padding='post').astype('int32')
        next_w_char = np.reshape(next_w_char, (next_w_char.shape[0], 1, next_w_char.shape[1]))  # batch x 1 x char_len
        chosen.append(choose_this_one)
        chosen_probs.append(probs)
    chosen = np.array(chosen)  # time x batch
    chosen = np.transpose(chosen, (1, 0))  # batch x time
    chosen_probs = torch.stack(chosen_probs, dim=1)  # batch x time

    # mask things after </s> out
    res_mask = np.zeros(chosen.shape, dtype='float32')
    res_list = []
    longest = -1
    for i, c in enumerate(chosen):
        c = c.tolist()
        if eos_id in c:
            idx = c.index(eos_id)
            res_mask[i, :idx + 1] = 1.0
            c = c[:idx]
        else:
            res_mask[i, :] = 1.0
        if len(c) > longest:
            longest = len(c)
        res_list.append(c)
    chosen_probs = chosen_probs[:, :longest]
    res_mask = res_mask[:, :longest]
    res_mask = to_pt(res_mask, enable_cuda=enable_cuda, type="float")
    return res_list, chosen_probs, res_mask


def sample_a_batch(_model, source, source_char, local_dict, id2word, char2id,
                   sample=False, generate_this_many_peyphrases=5, max_keyphrase_length=5, beam_size=5, enable_cuda=False):
    if isinstance(source, np.ndarray):
        source = to_pt(source, enable_cuda=enable_cuda)
    if isinstance(source_char, np.ndarray):
        source_char = to_pt(source_char, enable_cuda=enable_cuda)
    assert len(local_dict) == source.size(0)  # same batch size
    assert generate_this_many_peyphrases > 0
    assert max_keyphrase_length > 0
    local_id2word = [{v: k for k, v in d.items()} for d in local_dict]

    history_cache = [[] for i in range(len(local_dict))]
    res_word_ids, res_word_strings = [[] for i in range(len(local_dict))], [[] for i in range(len(local_dict))]
    for ith_keyphrase in range(generate_this_many_peyphrases):
        if ith_keyphrase == 0:
            # 1st generation, history is empty
            history = np.zeros((len(local_dict), 1), dtype='int32')
        else:
            history = pad_sequences(history_cache, padding='post').astype('int32')
        history_char = add_char_level_inputs(history, id2word, char2id, local_dict)

        history = to_pt(history, enable_cuda=enable_cuda)
        history_char = to_pt(history_char, enable_cuda=enable_cuda)

        sampled_word_ids, sampled_word_probs, word_probs_mask = sample_or_argmax((source, history, source_char, history_char, local_dict), _model,
                                                                                 id2word=id2word, char2id=char2id, max_sequence_length=max_keyphrase_length,
                                                                                 sample=sample, enable_cuda=enable_cuda)
        history_cache = [hist + new for hist, new in zip(history_cache, sampled_word_ids)]
        for j in range(len(res_word_ids)):
            res_word_ids[j].append(sampled_word_ids[j])
            word_string = []
            for wid in sampled_word_ids[j]:
                if wid < len(id2word):
                    word_string.append(id2word[wid])
                else:
                    word_string.append(local_id2word[wid])
            word_string = " ".join(word_string)
            res_word_strings[j].append(word_string)
    return res_word_ids, res_word_strings


# def _tile(inp, n):
#     if isinstance(inp, np.ndarray):
#         if len(inp.shape) == 2:
#             return np.tile(inp, (n, 1))
#         elif len(inp.shape) == 3:
#             return np.tile(inp, (n, 1, 1))
#         else:
#             raise NotImplemented
#     else:
#         if len(inp.size()) == 2:
#             return inp.repeat(n, 1)
#         elif len(inp.size()) == 3:
#             return inp.repeat(n, 1, 1)
#         else:
#             raise NotImplemented


# def beam_search_command_by_template(inputs, f_init, f_next, verb_mask, object_mask,
#                                     pad_id=0, unk_id=1, eos_id=2, bos_id=3,
#                                     id2word=None, char2id=None, single_word_id=[],
#                                     beam_size=3, sample=False, enable_cuda=False):
#     chosen = []
#     description, inventory, feedback, quest, description_char, inventory_char, feedback_char, quest_char, prev_game_info = inputs
#     batch_size = description.size(0)
#     vocab_size = len(id2word)
#     # use f_init to get decoder init states and source / condition encodings
#     description_encoding_sequence, description_mask, inventory_encoding_sequence, inventory_mask,\
#         feedback_encoding_sequence, feedback_mask, quest_encoding_sequence, quest_mask, next_h, next_c = f_init(description, inventory, feedback, quest, description_char, inventory_char, feedback_char, quest_char)

#     # beginning of sentence indicator is bos_id
#     next_w = np.ones((batch_size, 1), dtype='int32') * bos_id
#     hyp_scores = np.zeros((batch_size,), dtype='float32')

#     _temp_char_id_list = [char2id[ch] for ch in list('<S>')]
#     next_w_char = np.array([_temp_char_id_list] * batch_size).astype('int32')
#     next_w_char = np.reshape(next_w_char, (batch_size, 1, next_w_char.shape[-1]))
#     # first_object_cache = None
#     hyp_samples = [[[]]]
#     sequence_len = None

#     for t in range(3):
#         prev_word_ids = copy.deepcopy(next_w[:, 0])
#         next_w = to_pt(next_w, enable_cuda=enable_cuda)
#         next_w_char = to_pt(next_w_char, enable_cuda=enable_cuda)
#         live_beam = next_h.size(0) // batch_size
#         if t > 0:
#             assert beam_size == live_beam

#         # get next states
#         if t == 0:
#             temp_vocab_mask = copy.deepcopy(verb_mask)
#         elif t == 1:
#             temp_vocab_mask = copy.deepcopy(object_mask)
#         else:
#             temp_vocab_mask = copy.deepcopy(object_mask)
#             temp_vocab_mask[:, pad_id] = 1.0
#             temp_vocab_mask[:, eos_id] = 1.0
#         temp_vocab_mask = _tile(temp_vocab_mask, live_beam)  # batch*beam x vocab
#         temp_vocab_mask[np.arange(temp_vocab_mask.shape[0]), prev_word_ids] = 0.0
#         # non_zeros = np.min(np.sum(temp_vocab_mask, 1))
#         # assert non_zeros >= live_beam
#         vocab_mask = to_pt(temp_vocab_mask, enable_cuda=enable_cuda, type='float')  # batch x vocab_size
#         des = _tile(description_encoding_sequence, live_beam)  # batch*beam x description_len x enc
#         dm = _tile(description_mask, live_beam)  # batch*beam x description_len
#         dd = _tile(description, live_beam)  # batch*beam x description_len
#         ies = _tile(inventory_encoding_sequence, live_beam)  # batch*beam x inventory_len x enc
#         im = _tile(inventory_mask, live_beam)  # batch*beam x inventory_len
#         ii = _tile(inventory, live_beam)  # batch*beam x inventory_len
#         fes = _tile(feedback_encoding_sequence, live_beam)  # batch*beam x feedback_len x enc
#         fm = _tile(feedback_mask, live_beam)  # batch*beam x feedback_len
#         ff = _tile(feedback, live_beam)  # batch*beam x feedback_len
#         qes = _tile(quest_encoding_sequence, live_beam)  # batch*beam x quest_len x enc
#         qm = _tile(quest_mask, live_beam)  # batch*beam x quest_len
#         qq = _tile(quest, live_beam)  # batch*beam x quest_len
#         pgi = _tile(prev_game_info, live_beam)  # batch*beam x dec

#         p_positions_mapped, p_target_vocab, next_h, next_c = f_next(next_w, next_w_char,
#                                                                     dd, des, dm,
#                                                                     ii, ies, im,
#                                                                     ff, fes, fm,
#                                                                     qq, qes, qm,
#                                                                     pgi, next_h, next_c)
#         pred = p_target_vocab
#         for p in p_positions_mapped:
#             pred = pred + p
#         pred = mask_distributions(pred, vocab_mask)

#         log_pred = torch.log(pred)  # * gt_zero  # batch*beam x time
#         log_pred = log_pred.cpu().data.numpy()  # batch*beam x vocab_size
#         next_h = next_h.view(-1, batch_size, next_h.size(-1)).permute(1, 0, 2).contiguous()  # batch x beam x hid
#         next_c = next_c.view(-1, batch_size, next_c.size(-1)).permute(1, 0, 2).contiguous()  # batch x beam x hid

#         cand_scores = hyp_scores[:, None] - log_pred  # batch*beam x vocab size
#         if sequence_len is None:
#             _norm_cand_scores = cand_scores
#         else:
#             sequence_len[sequence_len == 0.0] = 1.0
#             _norm_cand_scores = cand_scores / (sequence_len[:, None] + 1.0)

#         cand_scores = np.reshape(cand_scores, (-1, batch_size, vocab_size))  # beam x batch x vocab
#         cand_scores = np.transpose(cand_scores, (1, 0, 2))  # batch x beam x vocab
#         cand_scores = np.reshape(cand_scores, (batch_size, -1))  # batch x vocab*beam

#         _norm_cand_scores = np.reshape(_norm_cand_scores, (-1, batch_size, vocab_size))  # beam x batch x vocab
#         _norm_cand_scores = np.transpose(_norm_cand_scores, (1, 0, 2))  # batch x beam x vocab
#         _norm_cand_scores = np.reshape(_norm_cand_scores, (batch_size, -1))  # batch x vocab*beam

#         if sample:
#             # sample from top 2*live_beam probabilities
#             _csc = copy.deepcopy(_norm_cand_scores)
#             # non_infs = _csc[_csc != np.inf]
#             # _min = np.min(non_infs, -1, keepdims=True)
#             # _max = np.max(non_infs, -1, keepdims=True)
#             # _csc[_csc != np.inf] -= _min
#             # _csc[_csc != np.inf] /= (_max - _min)  # [0, 1]
#             # _csc[_csc != np.inf] -= 0.5  # [-.5, .5]
#             # _csc[_csc != np.inf] *= 10  # [-5, 5]
#             cand_exp = np.exp(-_csc)
#             samp_prob = cand_exp / np.sum(cand_exp, -1, keepdims=True)  # batch x vocab*beam
#             # sample live_beam out of top 2*live_beam
#             # sample_from_this_many = min(max(beam_size * 2, 100), samp_prob.shape[1])  # TODO: 100
#             # amax = (-samp_prob).argpartition(sample_from_this_many - 1, axis=-1)[:, :sample_from_this_many]

#             ranks_flat = []
#             for i in range(batch_size):
#                 # _prob = samp_prob[i][amax[i]]
#                 # _prob /= _prob.sum()
#                 # attention: the amount of non-zero element in _prob (non-zero in vocab mask) should be larger than beam size
#                 non_0 = np.sum((samp_prob[i] != 0.0).astype('float32'))
#                 if non_0 >= beam_size:
#                     _rf = np.random.choice(samp_prob.shape[1], beam_size, replace=False, p=samp_prob[i])
#                 else:
#                     _rf = _norm_cand_scores[i].argpartition(beam_size - 1)[:beam_size]
#                 ranks_flat.append(_rf)
#             ranks_flat = np.array(ranks_flat)
#         else:
#             ranks_flat = _norm_cand_scores.argpartition(beam_size - 1, axis=-1)[:, :beam_size]  # idx of live_beam max values in cand_scores
#         costs = np.array([np.take(cs, rf) for cs, rf in zip(cand_scores, ranks_flat)])  # batch x beam_size

#         # Find out to which initial hypothesis idx this was belonging
#         # Find out the idx of the appended word
#         trans_indices = ranks_flat // vocab_size  # batch x beam
#         word_indices = ranks_flat % vocab_size  # batch x beam
#         new_h, new_c = [], []
#         new_hyp_scores = []
#         new_hyp_samples = []
#         sequence_len_update = []

#         for one_in_a_batch in range(batch_size):

#             tmp_scores = []
#             tmp_samples = []
#             tmp_sequence_len = []
#             # This will be the new next states in the next iteration
#             hyp_h = []
#             hyp_c = []
#             # Iterate over the hypotheses and add them to new_* lists
#             trans_idx, word_idx = trans_indices[one_in_a_batch], word_indices[one_in_a_batch]
#             for idx, [ti, wi] in enumerate(zip(trans_idx, word_idx)):
#                 # Form the new hypothesis by appending new word to the left hyp
#                 if len(hyp_samples) <= one_in_a_batch or len(hyp_samples[one_in_a_batch]) <= ti or len(hyp_samples[one_in_a_batch][ti]) == 0:
#                     new_hyp = [wi]
#                 else:
#                     new_hyp = hyp_samples[one_in_a_batch][ti] + [wi]

#                 # Add formed hypothesis to the new hypotheses list
#                 tmp_samples.append(new_hyp)
#                 # Cumulated cost of this hypothesis
#                 if wi in [eos_id, pad_id]:
#                     tmp_scores.append(0)
#                     tmp_sequence_len.append(0.0)
#                 elif len(hyp_samples) <= one_in_a_batch or len(hyp_samples[one_in_a_batch]) <= ti or len(hyp_samples[one_in_a_batch][ti]) == 0:
#                     tmp_scores.append(costs[one_in_a_batch][idx])
#                     tmp_sequence_len.append(1.0)
#                 elif hyp_samples[one_in_a_batch][ti][-1] in [eos_id, pad_id] + single_word_id:
#                     tmp_scores.append(0)
#                     tmp_sequence_len.append(0.0)
#                 elif len(hyp_samples[one_in_a_batch][ti]) > 1 and hyp_samples[one_in_a_batch][ti][-2] in [eos_id, pad_id] + single_word_id:
#                     tmp_scores.append(0)
#                     tmp_sequence_len.append(0.0)
#                 else:
#                     tmp_scores.append(costs[one_in_a_batch][idx])
#                     tmp_sequence_len.append(1.0)
#                 # Hidden state of the decoder for this hypothesis
#                 hyp_h.append(next_h[one_in_a_batch][ti])
#                 hyp_c.append(next_c[one_in_a_batch][ti])
#             new_hyp_scores.append(np.array(tmp_scores, dtype='float32'))  # list of beam
#             sequence_len_update.append(np.array(tmp_sequence_len, dtype='float32'))  # list of beam
#             new_hyp_samples.append(tmp_samples)
#             new_h.append(torch.stack(hyp_h, 0))
#             new_c.append(torch.stack(hyp_c, 0))
#         next_h = torch.stack(new_h, 0)  # batch x beam x hid
#         next_h = next_h.permute(1, 0, 2).contiguous().view(next_h.size(0) * next_h.size(1), next_h.size(2))  # batch*beam x hid
#         next_c = torch.stack(new_c, 0)  # batch x beam x hid
#         next_c = next_c.permute(1, 0, 2).contiguous().view(next_c.size(0) * next_c.size(1), next_c.size(2))  # batch*beam x hid
#         hyp_scores = np.stack(new_hyp_scores, axis=0)  # batch x beam
#         hyp_scores = hyp_scores.T.reshape(-1)  # batch*beam
#         hyp_samples = new_hyp_samples  # list: batch, beam, []
#         sequence_len_update = np.stack(sequence_len_update, axis=0)  # batch x beam
#         sequence_len_update = sequence_len_update.T.reshape(-1)  # batch*beam
#         if sequence_len is None:
#             sequence_len = sequence_len_update
#         else:
#             sequence_len += sequence_len_update

#         next_w = []
#         for _i in range(len(hyp_samples[0])):
#             for batch_sample in hyp_samples:
#                 next_w.append(batch_sample[_i][-1])
#         sampled_words = [id2word[w] for w in next_w]  # batch of words
#         next_w_char = [list(w.lower()) for w in sampled_words]  # list of list of chars
#         next_w_char = [[char2id[ch] for ch in word] for word in next_w_char]  # list of list of char_ids
#         next_w_char = pad_sequences(next_w_char, maxlen=max_len(next_w_char), padding='post').astype('int32')
#         next_w_char = np.reshape(next_w_char, (next_w_char.shape[0], 1, next_w_char.shape[1]))  # batch x 1 x char_len
#         next_w = np.array(next_w).reshape((-1, 1))  # batch*beam x 1

#     # after 3 time steps
#     hyp_scores = np.reshape(hyp_scores, (batch_size, -1))  # batch x beam
#     sequence_len = np.reshape(sequence_len, (batch_size, -1))  # batch x beam
#     sequence_len[sequence_len <= 0] = 1.0
#     hyp_scores /= sequence_len  # batch x beam
#     which_beam = np.argmin(hyp_scores, -1)  # batch

#     res_list = []
#     for one_in_a_batch in range(batch_size):
#         chosen = hyp_samples[one_in_a_batch][which_beam[one_in_a_batch]]  # list

#         if eos_id in chosen:
#             i = chosen.index(eos_id)
#             chosen = chosen[:i]
#         if pad_id in chosen:
#             i = chosen.index(pad_id)
#             chosen = chosen[:i]
#         for sw in single_word_id:
#             if sw in chosen:
#                 i = chosen.index(sw) + 1
#                 chosen = chosen[:i]
#                 break

#         res_list.append(chosen)
#     return res_list
