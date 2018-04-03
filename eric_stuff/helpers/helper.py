# coding=utf-8
import numpy as np
import torch
try:
    import queue
except ImportError:
    import Queue as queue
import threading
import time


def print_data_samples(dataset, head, tail, id2word):
    for num in range(head, tail):
        print("================================= data num {} =================================".format(num))
        local_id2word = {v: k for k, v in dataset['local_oov_dict'][num].items()}

        print('---------------------------------------          source with oov words')
        tmp = []
        for a in dataset['input_source'][num]:
            if a == 0:
                continue
            if a < 50000:
                tmp.append(id2word[a])
            else:
                tmp.append(local_id2word[a])
        print(' '.join(tmp))
        print('---------------------------------------          target with oov words')
        tmp = []
        for a in dataset['input_target'][num]:
            if a == 0:
                continue
            if a < 50000:
                tmp.append(id2word[a])
            else:
                tmp.append(local_id2word[a])
        print(' '.join(tmp))
        print('---------------------------------------          prev target with oov words')
        tmp = []
        for a in dataset['input_prev_target'][num]:
            if a == 0:
                continue
            if a < 50000:
                tmp.append(id2word[a])
            else:
                tmp.append(local_id2word[a])
        print(' '.join(tmp))
        print('---------------------------------------          local oov dict')
        print(dataset['local_oov_dict'][num])


def torch_model_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + ' (\n'
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_model_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = torch.nn.modules.module._addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr += ', parameters={}'.format(params)
        tmpstr += '\n'

    tmpstr = tmpstr + ')'
    return tmpstr


def shuffle_data_dict(data_dict):
    '''
    Shuffle each data array in the named array dict.
    -- Assume all input arrays have same array.shape[0].
    '''
    ary_len = list(data_dict.values())[0].shape[0]
    rand_perm = np.random.permutation(ary_len)
    for k, v in data_dict.items():
        # permute input_dict[k]
        data_dict[k] = v.take(rand_perm, axis=0)
    return data_dict


def max_len(list_of_list):
    return max(map(len, list_of_list))


def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.):
    '''
    FROM KERAS
    Pads each sequence to the same length:
    the length of the longest sequence.
    If maxlen is provided, any sequence longer
    than maxlen is truncated to maxlen.
    Truncation happens off either the beginning (default) or
    the end of the sequence.
    Supports post-padding and pre-padding (default).
    # Arguments
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        padding: 'pre' or 'post', pad either before or after each sequence.
        truncating: 'pre' or 'post', remove values from sequences larger than
            maxlen either in the beginning or in the end of the sequence
        value: float, value to pad the sequences to the desired value.
    # Returns
        x: numpy array with dimensions (number_of_sequences, maxlen)
    '''
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def pad_sequences_of_sequences(sequences, pad=0.):
    # input: list of list of list
    max_dim_1 = max_len(sequences)
    max_dim_2 = max([max_len(s) for s in sequences])
    A = pad * np.ones((len(sequences), max_dim_1, max_dim_2), dtype='int32')
    for i in range(len(sequences)):
        for j in range(len(sequences[i])):
            for k in range(len(sequences[i][j])):
                    A[i][j][k] = sequences[i][j][k]
    return A


def trim_batch(batch, trim_margin=None):
    # for post padding
    # batch.shape: N * n_words
    if trim_margin is None:

        batch_temp = batch[:, ::-1]
        batch_temp = np.cumsum(batch_temp, axis=1)
        batch_temp = batch_temp[:, ::-1]
        zero = batch_temp == 0
        z_index = np.argmax(zero, axis=1)
        trim_margin = np.max(z_index)
    return batch[:, :trim_margin + 1], trim_margin


def trim(batch_dict):
    batch_dict['input_source'], _ = trim_batch(batch_dict['input_source'])
    batch_dict['input_prev_target'], _ = trim_batch(batch_dict['input_prev_target'])
    target = batch_dict['input_target']
    batch_dict['input_target'], _ = trim_batch(np.concatenate([np.ones((target.shape[0], 1), dtype='int32') * 1, target], axis=1))  # <s> + target
    batch_dict['output_target'], _ = trim_batch(np.concatenate([np.ones((target.shape[0], 1), dtype='int32') * 2, target], axis=1))  # target + </s>
    batch_dict['output_target_mask'] = (batch_dict['output_target'] > 0).astype('int32')

    return batch_dict


def random_generator(data_dict, input_keys, output_keys, batch_size, bucket_size=-1, sort_by=None, trim_function=None,
                     id2word=None, char2id=None, random_shuffle=True, enable_cuda=False):
    if bucket_size == -1:
        bucket_size = batch_size * 100
    sample_count = None
    for k, v in data_dict.items():
        if sample_count is None:
            sample_count = v.shape[0]
        if not (sample_count == v.shape[0]):
            raise Exception('Mismatched sample counts in data_dict.')

    if bucket_size > sample_count:
        bucket_size = sample_count
        print('bucket_size < sample_count')
    # epochs discard dangling samples that won't fill a bucket.
    buckets_per_epoch = sample_count // bucket_size
    if sample_count % bucket_size > 0:
        buckets_per_epoch += 1

    while True:
        # random shuffle
        if random_shuffle:
            data_dict = shuffle_data_dict(data_dict)

        for bucket_num in range(buckets_per_epoch):
            # grab the chunk of samples in the current bucket
            bucket_start = bucket_num * bucket_size
            bucket_end = bucket_start + bucket_size
            if bucket_start >= sample_count:
                continue
            bucket_end = min(bucket_end, sample_count)
            current_bucket_size = bucket_end - bucket_start
            bucket_idx = np.arange(bucket_start, bucket_end)
            bucket_dict = {k: v.take(bucket_idx, axis=0) for k, v in data_dict.items()}

            if sort_by is not None:
                non_zero = bucket_dict[sort_by]
                while non_zero.ndim > 2:
                    non_zero = np.max(non_zero, axis=-1)
                pad_counts = np.sum((non_zero == 0), axis=1)
                sort_idx = np.argsort(pad_counts)
                bucket_dict = {k: v.take(sort_idx, axis=0) for k, v in bucket_dict.items()}

            batches_per_bucket = current_bucket_size // batch_size
            if current_bucket_size % batch_size > 0:
                batches_per_bucket += 1

            for batch_num in range(batches_per_bucket):
                # grab the chunk of samples in the current bucket
                batch_start = batch_num * batch_size
                batch_end = batch_start + batch_size
                if batch_start >= current_bucket_size:
                    continue
                batch_end = min(batch_end, current_bucket_size)
                batch_idx = np.arange(batch_start, batch_end)
                batch_dict = {k: v.take(batch_idx, axis=0) for k, v in bucket_dict.items()}
                if trim_function is not None:
                    batch_dict = trim_function(batch_dict)
                batch_dict['input_source_char'] = add_char_level_inputs(batch_dict['input_source'], id2word, char2id)
                batch_dict['input_target_char'] = add_char_level_inputs(batch_dict['input_target'], id2word, char2id)
                batch_dict['input_prev_target_char'] = add_char_level_inputs(batch_dict['input_prev_target'], id2word, char2id)

                if enable_cuda:
                    batch_data = [torch.autograd.Variable(torch.from_numpy(batch_dict[k]).type(torch.LongTensor).cuda()) for k in input_keys] +\
                                 [torch.autograd.Variable(torch.from_numpy(batch_dict[k]).type(torch.LongTensor).cuda()) for k in output_keys]
                else:
                    batch_data = [torch.autograd.Variable(torch.from_numpy(batch_dict[k]).type(torch.LongTensor)) for k in input_keys] +\
                                 [torch.autograd.Variable(torch.from_numpy(batch_dict[k]).type(torch.LongTensor)) for k in output_keys]
                yield batch_data


def add_char_level_inputs(input_batch=None, id2word=None, char2id=None):
    if input_batch is None or id2word is None or char2id is None:
        return None
    flag = False
    input_batch = input_batch.astype('int32')
    if len(input_batch.shape) == 3:
        # batch x cmd x time
        shape0, shape1 = input_batch.shape[0], input_batch.shape[1]
        input_batch = np.reshape(input_batch, (-1, input_batch.shape[2]))
        flag = True

    word_set = set()
    for item in input_batch:
        word_set |= set([id2word[w] for w in item])
    word_set = list(word_set)
    max_char_in_word = max(map(len, word_set))

    char_matrix = np.zeros((input_batch.shape[0], input_batch.shape[1], max_char_in_word), dtype='int32')
    for i in range(input_batch.shape[0]):
        for j in range(input_batch.shape[1]):
            if input_batch[i][j] == 0:
                continue
            # it's actual word
            _w = id2word[input_batch[i][j]]
            _w = _w.lower()
            for k in range(len(_w)):
                try:
                    char_matrix[i][j][k] = char2id[_w[k]]
                except KeyError:
                    pass
    if flag:
        char_matrix = np.reshape(char_matrix, (shape0, shape1) + char_matrix.shape[1:])
    return char_matrix


def generator_queue(generator, max_q_size=10, wait_time=0.05, nb_worker=1):
    '''Builds a threading queue out of a data generator.
    Used in `fit_generator`, `evaluate_generator`.
    '''
    q = queue.Queue()
    _stop = threading.Event()

    def data_generator_task():
        while not _stop.is_set():
            try:
                if q.qsize() < max_q_size:
                    try:
                        generator_output = next(generator)
                    except ValueError:
                        continue
                    q.put(generator_output)
                else:
                    time.sleep(wait_time)
            except Exception:
                _stop.set()
                raise

    generator_threads = [threading.Thread(target=data_generator_task)
                         for _ in range(nb_worker)]

    for thread in generator_threads:
        thread.daemon = True
        thread.start()

    return q, _stop
