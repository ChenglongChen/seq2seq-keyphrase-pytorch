import time
import os
import yaml
import argparse
import torch
import math
from os.path import join as pjoin
import numpy as np
from tqdm import tqdm
from helpers.datasets import load_dataset
from helpers.helper import print_data_samples, torch_model_summarize, random_generator, trim, generator_queue
from helpers.model import CascadingGenerator, StandardNLL, DataParallel
from helpers.evaluation import eval_teacher_forcing, eval_free_running

wait_time = 0.01  # in seconds


def train_teacher_forcing(model_config):

    data_path = model_config['general']['data_path']
    _, id2word, char2id, id2char, train_data, teacher_forcing_valid_data, free_running_valid_data, test_data = load_dataset(data_path + "kp20k")
    id2word = id2word[:50000]
    print("there're {} words in word vocab, {} items in char vocab, train data size {}, valid data size {} / {}, test data size {}.".format(len(id2word), len(id2char), len(train_data['input_source']), len(teacher_forcing_valid_data['input_source']), len(free_running_valid_data['input_source']), len(test_data['input_source'])))
    if False:
        print('----------------------------------  print some data for debugging purpose')
        print_data_samples(train_data, 22, 35, id2word)
        exit(0)

    # Set the random seed manually for reproducibility.
    enable_cuda = model_config['general']['enable_cuda']
    torch.manual_seed(model_config['general']['torch_seed'])
    if torch.cuda.is_available():
        if not enable_cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(model_config['general']['cuda_seed'])

    # model
    _model = CascadingGenerator(model_config=model_config, word_vocab=id2word, char_vocab=id2char, word_vocab_oov_ext_size=1000, enable_cuda=enable_cuda)
    para_model = DataParallel(_model, device_ids=[0, 1, 2, 3])

    if enable_cuda:
        _model.cuda()

    criterion = StandardNLL()
    if enable_cuda:
        criterion = criterion.cuda()

    print('finished loading models')
    print(torch_model_summarize(_model))

    # get optimizer / lr
    init_learning_rate = model_config['optimizer']['learning_rate']
    parameters = filter(lambda p: p.requires_grad, _model.parameters())
    if model_config['optimizer']['step_rule'] == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=init_learning_rate)
    elif model_config['optimizer']['step_rule'] == 'adam':
        optimizer = torch.optim.Adam(parameters, lr=init_learning_rate)

    batch_size = model_config['scheduling']['batch_size']
    valid_batch_size = model_config['scheduling']['valid_batch_size']

    input_keys = ['input_source', 'input_target', 'input_prev_target', 'input_source_char', 'input_target_char', 'input_prev_target_char']
    output_keys = ['output_target', 'output_target_mask']
    special_keys = ['local_oov_dict']
    train_batch_generator = random_generator(data_dict=train_data, input_keys=input_keys, output_keys=output_keys, special_keys=special_keys, batch_size=batch_size,
                                             trim_function=trim, sort_by='input_source',
                                             id2word=id2word, char2id=char2id,
                                             enable_cuda=enable_cuda)

    valid_batch_generator = random_generator(data_dict=teacher_forcing_valid_data, input_keys=input_keys, output_keys=output_keys, special_keys=special_keys, batch_size=valid_batch_size,
                                             trim_function=trim, sort_by='input_source',
                                             id2word=id2word, char2id=char2id,
                                             enable_cuda=enable_cuda)
    # train
    number_batch = (train_data['input_source'].shape[0] + batch_size - 1) // batch_size
    data_queue, _ = generator_queue(train_batch_generator, max_q_size=20)
    learning_rate = init_learning_rate
    best_val_ppl = None
    be_patient = 0

    # load model, if any
    try:
        save_f = open(model_config['scheduling']['model_checkpoint_path'], 'rb')
        _model = torch.load(save_f)
        print("loading best model------------------------------------------------------------------\n")
        # eval on valid set
        val_loss, val_ppl = eval_teacher_forcing(_model=para_model, batch_generator=valid_batch_generator,
                                                 number_batch=(teacher_forcing_valid_data['input_source'].shape[0] + valid_batch_size - 1) // valid_batch_size,
                                                 criterion=criterion)
        print("In teacher forcing evaluation, loss=%.5f, ppl=%.5f" % (val_loss, val_ppl))
        val_f1 = eval_free_running(_model=para_model, data=free_running_valid_data, batch_size=valid_batch_size, id2word=id2word, char2id=char2id, enable_cuda=enable_cuda)
        print("In free running evaluation, f1 score=%.5f" % (val_f1))

    except:
        pass

    try:
        for epoch in range(model_config['scheduling']['epoch']):
            # negative log likelihood learning
            para_model.train()
            sum_loss = 0.0
            with tqdm(total=number_batch, leave=True, ncols=180, ascii=True) as pbar:
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

                    optimizer.zero_grad()
                    para_model.zero_grad()

                    history_info = para_model.get_history_info(input_prev_target, input_prev_target_char)
                    p_positions_mapped, p_target_vocab, _, _ = para_model.forward(input_source, input_target, input_source_char, input_target_char, history_info)

                    preds = p_target_vocab
                    for p in p_positions_mapped:
                        preds = preds + p
                    preds = preds * output_target_mask.float().unsqueeze(-1)  # batch x time x vocab_size
                    loss = criterion(preds, output_target, output_target_mask)  # batch
                    loss = torch.mean(loss)  # 1
                    loss.backward()
                    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                    torch.nn.utils.clip_grad_norm(para_model.parameters(), model_config['optimizer']['clip_grad_norm'])
                    optimizer.step()  # apply gradients

                    batch_loss = loss.cpu().data.numpy()
                    sum_loss += batch_loss * batch_size
                    pbar.set_description('epoch=%d, negative log likelihood training batch=%d, avg_loss=%.5f, batch_loss=%.5f, batch_ppl=%.5f, lr=%.6f' % (epoch, i,
                                                                                                                                                           sum_loss / float(batch_size * (i + 1)),
                                                                                                                                                           batch_loss,
                                                                                                                                                           math.exp(batch_loss) if batch_loss < 13. else np.inf,
                                                                                                                                                           learning_rate))
                    pbar.update(1)

            # eval on valid set
            val_loss, val_ppl = eval_teacher_forcing(_model=para_model, batch_generator=valid_batch_generator,
                                                     number_batch=(teacher_forcing_valid_data['input_source'].shape[0] + valid_batch_size - 1) // valid_batch_size,
                                                     criterion=criterion)
            # soft_val_f1, hard_val_f1 = evaluate_free_running(model=_model, data_generator=valid_batch_generator, data_size=valid_data['input_description'].shape[0],
            #                                                  id2word=word_vocab, char2id=char2id, batch_size=valid_batch_size, enable_cuda=enable_cuda)
            print("epoch=%d, in teacher forcing evaluation, loss=%.5f, ppl=%.5f" % (epoch, val_loss, val_ppl))
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_ppl or val_ppl < best_val_ppl:
                with open(model_config['scheduling']['model_checkpoint_path'], 'wb') as save_f:
                    torch.save(para_model, save_f)
                best_val_ppl = val_ppl
                be_patient = 0
            else:
                if epoch >= model_config['optimizer']['learning_rate_decay_from_this_epoch']:
                    if be_patient >= model_config['optimizer']['learning_rate_decay_patience']:
                        if learning_rate * model_config['optimizer']['learning_rate_decay_ratio'] > model_config['optimizer']['learning_rate_cut_lowerbound'] * model_config['optimizer']['learning_rate']:
                            # Anneal the learning rate if no improvement has been seen in the validation dataset.
                            print('cutting learning rate from %.5f to %.5f' % (learning_rate, learning_rate * model_config['optimizer']['learning_rate_decay_ratio']))
                            learning_rate *= model_config['optimizer']['learning_rate_decay_ratio']
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = learning_rate
                        else:
                            print('learning rate %.5f reached lower bound' % (learning_rate))
                    be_patient += 1

    # At any point you can hit Ctrl + C to break out of training early.
    except KeyboardInterrupt:
        print('--------------------------------------------\n')
        print('Exiting from training early\n')


if __name__ == '__main__':
    for _p in ['saved_models']:
        if not os.path.exists(_p):
            os.mkdir(_p)
    parser = argparse.ArgumentParser(description="train network.")
    parser.add_argument("-c", "--config_dir", default='config', help="the default config directory")
    args = parser.parse_args()

    # Read config from yaml file.
    config_file = pjoin(args.config_dir, 'config.yaml')
    with open(config_file) as reader:
        config = yaml.safe_load(reader)
    train_teacher_forcing(model_config=config)
