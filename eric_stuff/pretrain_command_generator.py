import time
import torch
import math
import numpy as np
from tqdm import tqdm
from helpers.datasets import CMDGeneratorDataset
from helpers.model import PointerSoftmaxNLL
from helpers.lm_helper import print_shape_info, print_data_samples, random_generator, lm_trim, torch_model_summarize, generator_queue, evaluate
from helpers.lm_helper import evaluate_free_running

wait_time = 0.01  # in seconds


def pretrain_command_generator_model(_model, model_config, word_vocab, char_vocab):

    enable_cuda = model_config['scheduling']['enable_cuda']
    dataset = CMDGeneratorDataset(dataset_h5=model_config['dataset']['h5'],
                                  data_path='command_lm_dataset.0.3/',
                                  predefined_word_vocab=word_vocab,
                                  use_template=True)

    train_data, valid_data = dataset.get_data()
    print_shape_info(train_data)
    if False:
        print('----------------------------------  printing out data shape')
        print_data_samples(dataset, train_data, 12, 15)
        exit(0)

    # Set the random seed manually for reproducibility.
    torch.manual_seed(model_config['scheduling']['cuda_seed'])
    if torch.cuda.is_available():
        if not enable_cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(model_config['scheduling']['cuda_seed'])

    if enable_cuda:
        _model.cuda()

    criterion = PointerSoftmaxNLL()
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

    input_keys = ['input_description', 'input_inventory', 'input_action', 'input_prev_action', 'input_description_char', 'input_inventory_char', 'input_action_char', 'input_prev_action_char']
    output_keys = ['input_label', 'input_label_mask', 'input_switch_vocab', 'input_switch_source_position']
    batch_size = model_config['scheduling']['batch_size']
    valid_batch_size = model_config['scheduling']['valid_batch_size']
    char2id = {}
    for i, ch in enumerate(char_vocab):
        char2id[ch] = i

    train_batch_generator = random_generator(data_dict=train_data, batch_size=batch_size,
                                             input_keys=input_keys, output_keys=output_keys,
                                             trim_function=lm_trim, sort_by='input_action',
                                             id2word=word_vocab, char2id=char2id,
                                             enable_cuda=enable_cuda)

    valid_batch_generator = random_generator(data_dict=valid_data, batch_size=batch_size,
                                             input_keys=input_keys, output_keys=output_keys,
                                             trim_function=lm_trim, sort_by='input_action',
                                             id2word=word_vocab, char2id=char2id,
                                             enable_cuda=enable_cuda)
    # train
    number_batch = (train_data['input_description'].shape[0] + batch_size - 1) // batch_size
    data_queue, _ = generator_queue(train_batch_generator, max_q_size=20)
    learning_rate = init_learning_rate
    best_val_ppl = None
    be_patient = 0

    # load model, if any
    try:
        save_f = open(model_config['dataset']['model_save_path'], 'rb')
        _model = torch.load(save_f)
        # Run on test data.
        print("loading best model------------------------------------------------------------------\n")
        test_loss, test_ppl = evaluate(model=_model, data_generator=valid_batch_generator,
                                       data_size=valid_data['input_description'].shape[0], criterion=criterion,
                                       batch_size=valid_batch_size)

        soft_test_f1, hard_test_f1 = evaluate_free_running(model=_model, data_generator=valid_batch_generator, data_size=valid_data['input_description'].shape[0],
                                                           id2word=word_vocab, char2id=char2id, batch_size=valid_batch_size, enable_cuda=enable_cuda)
        print("------------------------------------------------------------------------------------\n")
        print("loss=%.5f, ppl=%.5f, soft f1=%.5f, hard f1=%.5f" % (test_loss, test_ppl, soft_test_f1, hard_test_f1))

    except:
        pass

    try:
        for epoch in range(model_config['scheduling']['epoch']):
            # negative log likelihood learning
            _model.train()
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
                    input_description, input_inventory, input_action, input_prev_action, input_description_char, input_inventory_char, input_action_char, input_prev_action_char,\
                        input_label, input_label_mask, switch_vocab, switch_source_position = generator_output

                    optimizer.zero_grad()
                    _model.zero_grad()
                    p_source_position, p_target_vocab = _model.forward(input_description, input_inventory, input_action, input_prev_action, input_description_char, input_inventory_char, input_action_char, input_prev_action_char)  # batch x action_time x vocab
                    # loss
                    loss = criterion(p_target_vocab, p_source_position, input_label, input_label_mask, switch_vocab, switch_source_position)  # batch
                    loss = torch.mean(loss)
                    loss.backward()
                    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                    torch.nn.utils.clip_grad_norm(_model.parameters(), model_config['optimizer']['clip_grad_norm'])
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
            val_loss, val_ppl = evaluate(model=_model, data_generator=valid_batch_generator,
                                         data_size=valid_data['input_description'].shape[0], criterion=criterion,
                                         batch_size=valid_batch_size)
            soft_val_f1, hard_val_f1 = evaluate_free_running(model=_model, data_generator=valid_batch_generator, data_size=valid_data['input_description'].shape[0],
                                                             id2word=word_vocab, char2id=char2id, batch_size=valid_batch_size, enable_cuda=enable_cuda)
            print("epoch=%d, valid loss=%.5f, valid ppl=%.5f, soft f1=%.5f, hard f1=%.5f, lr=%.6f" % (epoch, val_loss, val_ppl, soft_val_f1, hard_val_f1, learning_rate))
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_ppl or val_ppl < best_val_ppl:
                with open(model_config['dataset']['model_save_path'], 'wb') as save_f:
                    torch.save(_model, save_f)
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

    # Load the best saved model.
    with open(model_config['dataset']['model_save_path'], 'rb') as save_f:
        _model = torch.load(save_f)

        # Run on test data.
        print("loading best model------------------------------------------------------------------\n")
        test_loss, test_ppl = evaluate(model=_model, data_generator=valid_batch_generator,
                                       data_size=valid_data['input_description'].shape[0], criterion=criterion,
                                       batch_size=valid_batch_size)

        soft_test_f1, hard_test_f1 = evaluate_free_running(model=_model, data_generator=valid_batch_generator, data_size=valid_data['input_description'].shape[0],
                                                           word_vocab=word_vocab, batch_size=valid_batch_size, enable_cuda=enable_cuda)
        print("------------------------------------------------------------------------------------\n")
        print("loss=%.5f, ppl=%.5f, soft f1=%.5f, hard f1=%.5f" % (test_loss, test_ppl, soft_test_f1, hard_test_f1))

    return _model
