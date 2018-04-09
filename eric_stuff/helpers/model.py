import logging
import numpy as np

import torch
from helpers.helper import to_pt
from helpers.layers import Embedding, TimeDistributedEmbedding
from helpers.layers import TimeDistributedRNN
from helpers.layers import MergeDistributions
from helpers.layers import FastBiLSTM, BiLSTM, MultiMLPDecoder
from helpers.layers import MultiGenerateFromVocabulary, GenerativeMultiAttnDecoder, MultiPointerSoftmax

logger = logging.getLogger(__name__)


class StandardNLL(torch.nn.modules.loss._Loss):
    """
    Shape:
        y_pred:     batch x time x class
        y_true:     batch x time
        mask:       batch x time
        output:     batch
    """

    def forward(self, y_pred, y_true, mask):
        _eps = 1e-6
        mask = mask.float()
        P = torch.gather(y_pred.view(-1, y_pred.size(2)), 1, y_true.contiguous().view(-1, 1))  # batch*time x 1
        P = P.view(y_true.size(0), y_true.size(1))  # batch x time
        epsilon = torch.le(P, 0.0).float() * _eps  # batch x time
        log_P = torch.log(P + epsilon)  # * gt_zero  # batch x time
        log_P = log_P * mask  # batch x time
        sum_log_P = torch.sum(log_P, dim=1) / torch.sum(mask, dim=1)  # batch
        return -sum_log_P


class PointerSoftmaxNLL(torch.nn.modules.loss._Loss):
    """
    Shape:
        y_pred_vocab:                       batch x time x class
        y_pred_source_position (mapped):    batch x time x class
        y_true:                             batch x time
        mask:                               batch x time
        switch_vocab:                       batch x time
        switch_source_position:             batch x time
        output:                             batch
    """
    def log_likelihood(self, y_pred, y_true):
        _eps = 1e-6
        P = torch.gather(y_pred.view(-1, y_pred.size(2)), 1, y_true.contiguous().view(-1, 1))  # batch*time x 1
        P = P.view(y_true.size(0), y_true.size(1))  # batch x time
        epsilon = torch.le(P, 0.0).float() * _eps  # batch x time
        log_P = torch.log(P + epsilon)  # * gt_zero  # batch x time
        return log_P

    def forward(self, y_pred_vocab, y_pred_source_position, y_true, mask, switch_vocab, switch_source_position):
        mask = mask.float()
        switch_vocab = switch_vocab.float()
        switch_source_position = switch_source_position.float()
        log_P_vocab = self.log_likelihood(y_pred_vocab, y_true)  # batch x time
        log_P_source_position = self.log_likelihood(y_pred_source_position, y_true)  # batch x time
        log_P = log_P_vocab * switch_vocab + log_P_source_position * switch_source_position  # batch x time
        log_P = log_P * mask  # batch x time
        sum_log_P = torch.sum(log_P, dim=1) / torch.sum(mask, dim=1)  # batch
        return -sum_log_P


class PolicyGradientLoss(torch.nn.modules.loss._Loss):
    """
    Shape:
        y_pred_prob:    batch x time
        reward:         batch x time or batch
        mask:           batch x time
        output:         batch
    """

    def forward(self, y_pred, reward, mask=None):
        _eps = 1e-6
        epsilon = torch.le(y_pred, 0.0).float() * _eps  # batch x time
        log_P = torch.log(y_pred + epsilon)  # * gt_zero  # batch x time
        if len(reward.size()) == 1:
            reward = reward.unsqueeze(-1)
        log_P = log_P * reward  # batch x time
        if mask is not None:
            mask = mask.float()
            log_P = log_P * mask  # batch x time
        _loss = -torch.sum(log_P, dim=1)  # batch
        return _loss


class CascadingGenerator(torch.nn.Module):
    model_name = 'kp_generator'

    def __init__(self, model_config, word_vocab, char_vocab, word_vocab_oov_ext_size=0, enable_cuda=False):
        super(CascadingGenerator, self).__init__()
        self.model_config = model_config
        self.enable_cuda = enable_cuda
        self.word_vocab_size = len(word_vocab)
        self.char_vocab_size = len(char_vocab)
        self.id2word = word_vocab
        self.word_vocab_oov_ext_size = word_vocab_oov_ext_size
        self.read_config()
        self._def_layers()
        self.print_parameters()

    def print_parameters(self):
        amount = 0
        for p in self.parameters():
            amount += np.prod(p.size())
        print("total number of parameters: %s" % (amount))
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        amount = 0
        for p in parameters:
            amount += np.prod(p.size())
        print("number of trainable parameters: %s" % (amount))

    def read_config(self):
        # global config
        config = self.model_config[self.model_name]['global']
        self.fast_rnns = config['fast_rnns']
        self.dropout_between_rnn_hiddens = config['dropout_between_rnn_hiddens']
        self.dropout_between_rnn_layers = config['dropout_between_rnn_layers']
        self.dropout_in_rnn_weights = config['dropout_in_rnn_weights']
        self.use_layernorm = config['use_layernorm']
        self.use_highway_connections = config['use_highway_connections']

        # embedding config
        # word level
        config = self.model_config[self.model_name]['embedding']['word_level']
        self.embed_path = config['path']
        self.embedding_size = config['embedding_size']
        self.embedding_type = config['embedding_type']
        self.embedding_dropout = config['embedding_dropout']
        self.embedding_trainable = config['embedding_trainable']
        self.embedding_oov_init = config['embedding_oov_init']
        # char level
        config = self.model_config[self.model_name]['embedding']['char_level']
        self.char_embedding_size = config['embedding_size']
        self.char_embedding_rnn_size = config['embedding_rnn_size']
        self.char_embedding_dropout = config['embedding_dropout']
        self.char_embedding_trainable = config['embedding_trainable']

        # model config
        config = self.model_config[self.model_name][self.model_name]
        self.encoder_rnn_hidden_size = config['encoder_rnn_hidden_size']
        self.decoder_rnn_hidden_size = config['decoder_rnn_hidden_size']
        self.pointer_softmax_hidden_size = config['pointer_softmax_hidden_size']
        self.decoder_vocab_generator_hidden_size = config['decoder_vocab_generator_hidden_size']
        self.decoder_rnn_attention_size = config['decoder_rnn_attention_size']
        self.decoder_init_states_generator_hidden_size = config['decoder_init_states_generator_hidden_size']
        self.history_info_integrator_hidden_size = config['history_info_integrator_hidden_size']

    def _def_layers(self):

        # word embeddings
        self.word_embedding = Embedding(embedding_size=self.embedding_size,
                                        vocab_size=self.word_vocab_size,
                                        trainable=self.embedding_trainable,
                                        id2word=self.id2word,
                                        dropout_rate=self.embedding_dropout,
                                        embedding_oov_init=self.embedding_oov_init,
                                        embedding_type=self.embedding_type,
                                        pretrained_embedding_path=self.embed_path,
                                        vocab_oov_ext_size=self.word_vocab_oov_ext_size,
                                        enable_cuda=self.enable_cuda)

        # character embedding
        self.char_embedding = TimeDistributedEmbedding(Embedding(embedding_size=self.char_embedding_size,
                                                                 vocab_size=self.char_vocab_size,
                                                                 trainable=self.char_embedding_trainable,
                                                                 dropout_rate=self.char_embedding_dropout,
                                                                 embedding_type='random',
                                                                 enable_cuda=self.enable_cuda))
        if self.fast_rnns:
            char_encoder_rnn = FastBiLSTM(ninp=self.char_embedding_size,
                                          nhids=self.char_embedding_rnn_size,
                                          dropout_between_rnn_layers=self.dropout_between_rnn_layers)
        else:
            char_encoder_rnn = BiLSTM(nemb=self.char_embedding_size, nhids=self.char_embedding_rnn_size,
                                      dropout_between_rnn_hiddens=self.dropout_between_rnn_hiddens,
                                      dropout_between_rnn_layers=self.dropout_between_rnn_layers,
                                      dropout_in_rnn_weights=self.dropout_in_rnn_weights,
                                      use_layernorm=self.use_layernorm,
                                      use_highway_connections=self.use_highway_connections,
                                      enable_cuda=self.enable_cuda)

        self.char_encoder = TimeDistributedRNN(rnn=char_encoder_rnn)

        emb_output_size = self.embedding_size + self.char_embedding_rnn_size[-1]
        # lstm encoder
        if self.fast_rnns:
            self.encoder = FastBiLSTM(ninp=emb_output_size,
                                      nhids=self.encoder_rnn_hidden_size,
                                      dropout_between_rnn_layers=self.dropout_between_rnn_layers)
        else:
            self.encoder = BiLSTM(nemb=emb_output_size, nhids=self.encoder_rnn_hidden_size,
                                  dropout_between_rnn_hiddens=self.dropout_between_rnn_hiddens,
                                  dropout_between_rnn_layers=self.dropout_between_rnn_layers,
                                  dropout_in_rnn_weights=self.dropout_in_rnn_weights,
                                  use_layernorm=self.use_layernorm,
                                  use_highway_connections=self.use_highway_connections,
                                  enable_cuda=self.enable_cuda)

        self.history_info_integrator = MultiMLPDecoder(input_dim=[self.encoder_rnn_hidden_size[-1]],
                                                       hidden_dim=self.history_info_integrator_hidden_size, output_dim=self.encoder_rnn_hidden_size[-1], enable_cuda=self.enable_cuda)

        self.decoder_init_states_generator = MultiMLPDecoder(input_dim=[self.encoder_rnn_hidden_size[-1]],
                                                             hidden_dim=self.decoder_init_states_generator_hidden_size,
                                                             output_dim=self.decoder_rnn_hidden_size,
                                                             enable_cuda=self.enable_cuda)

        self.decoder = GenerativeMultiAttnDecoder(input_dim=emb_output_size, hidden_dim=self.decoder_rnn_hidden_size,
                                                  enc_dim=self.encoder_rnn_hidden_size[-1],
                                                  attn_dim=self.decoder_rnn_attention_size,
                                                  prev_info_dim=self.encoder_rnn_hidden_size[-1],
                                                  dropout_between_rnn_hiddens=self.dropout_between_rnn_hiddens, dropout_input=self.dropout_between_rnn_layers,
                                                  use_layernorm=self.use_layernorm, enable_cuda=self.enable_cuda, condition_amount=1)

        self.generate_from_vocab = MultiGenerateFromVocabulary(emb_dim=emb_output_size, dec_dim=self.decoder_rnn_hidden_size,
                                                               hidden_dim=self.decoder_vocab_generator_hidden_size,
                                                               enc_dim=self.encoder_rnn_hidden_size[-1],
                                                               ntokens=self.word_vocab_size + self.word_vocab_oov_ext_size, condition_amount=1)
        self.pointer_softmax = MultiPointerSoftmax(dec_dim=self.decoder_rnn_hidden_size, enc_dim=self.encoder_rnn_hidden_size[-1], hidden_dim=self.pointer_softmax_hidden_size, condition_amount=4)
        self.merge_distributions = MergeDistributions(enable_cuda=self.enable_cuda)

    def _embed(self, _input_words, _input_chars):
        # batch x time
        word_embedding, word_mask = self.word_embedding.forward(_input_words)  # batch x time x emb
        char_embedding, char_mask = self.char_embedding.forward(_input_chars)  # batch x time x char_len x emb
        _, char_embedding, _ = self.char_encoder.forward(char_embedding, char_mask)  # batch x time x emb
        embedding = torch.cat([word_embedding, char_embedding], -1)
        return embedding, word_mask

    def get_history_info(self, input_history, input_history_char):
        input_history_np = input_history.cpu().data.numpy()
        if np.sum(input_history_np) == 0:
            return to_pt(np.zeros((input_history_np.shape[0], self.encoder_rnn_hidden_size[-1])), enable_cuda=self.enable_cuda, type='float')

        # encode
        history_embeddings, history_mask = self._embed(input_history, input_history_char)  # batch x time x emb
        _, history_encoding, _ = self.encoder.forward(history_embeddings, history_mask)  # batch x hid_enc
        history_info = self.history_info_integrator.forward([history_encoding])  # batch x hid
        return history_info

    def _encode(self, input_source, input_source_char):
        # word embedding
        source_embeddings, source_mask = self._embed(input_source, input_source_char)  # batch x time x emb
        # encodings
        source_encoding_sequence, source_encoding, _ = self.encoder.forward(source_embeddings, source_mask)  # batch x hid_enc
        return source_encoding_sequence, source_encoding, source_mask

    def _decode(self, input_target, input_target_char, source_encoding_sequence, source_mask, history_info, init_states_h, init_states_c):
        target_embeddings, target_mask = self._embed(input_target, input_target_char)  # batch x time x emb
        hidden_states_h, hidden_states_c, weighted_encodings, attentions = self.decoder.forward(target_embeddings, target_mask, init_states_h, init_states_c,
                                                                                                [source_encoding_sequence],
                                                                                                [source_mask],
                                                                                                history_info)

        p_target_vocab = self.generate_from_vocab.forward(target_embeddings, target_mask, hidden_states_h, weighted_encodings)
        # batch x time x vocab_size
        p_positions, p_target_vocab, switch = self.pointer_softmax.forward(hidden_states_h, target_mask, weighted_encodings, attentions, p_target_vocab)
        return p_positions, p_target_vocab, hidden_states_h, hidden_states_c, target_mask, switch

    def forward(self, input_source, input_target, input_source_char, input_target_char, history_info):

        # encode
        source_encoding_sequence, source_encoding, source_mask = self._encode(input_source, input_source_char)

        # decoder init states
        init_states_h = self.decoder_init_states_generator.forward([source_encoding])  # batch x dec
        init_states_c = torch.autograd.Variable(torch.zeros(init_states_h.size()))
        if self.enable_cuda:
            init_states_c = init_states_c.cuda()

        # decode
        p_positions, p_target_vocab, _, _, target_mask, switch = self._decode(input_target, input_target_char, source_encoding_sequence, source_mask,
                                                                              history_info, init_states_h, init_states_c)
        p_positions_mapped = [self.merge_distributions(p, p_target_vocab, cond) for p, cond in zip(p_positions, [input_source])]

        return p_positions_mapped, p_target_vocab, target_mask, switch

    def f_init(self, input_source, input_source_char):
        # input_source: batch x time_source
        source_encoding_sequence, source_encoding, source_mask = self._encode(input_source, input_source_char)
        # decoder init states
        init_states_h = self.decoder_init_states_generator.forward([source_encoding])  # batch x dec
        init_states_c = torch.autograd.Variable(torch.zeros(init_states_h.size()))
        if self.enable_cuda:
            init_states_c = init_states_c.cuda()

        return source_encoding_sequence, source_mask, init_states_h, init_states_c

    def f_next(self, input_target, input_target_char,
               input_source, source_encoding_sequence, source_mask,
               history_info, init_states_h, init_states_c):

        p_positions, p_target_vocab, hidden_states_h, hidden_states_c, _, _ = self._decode(input_target, input_target_char, source_encoding_sequence, source_mask,
                                                                                           history_info, init_states_h, init_states_c)
        p_positions_mapped = [self.merge_distributions(p, p_target_vocab, cond) for p, cond in zip(p_positions, [input_source])]
        # RID SINGLETON TIME STEP FOR RECURRENCE
        h = hidden_states_h[:, 0]  # batch x hid
        c = hidden_states_c[:, 0]  # batch x hid
        p_positions_mapped = [p[:, 0] for p in p_positions_mapped]  # batch x vocab
        p_target_vocab = p_target_vocab[:, 0]  # batch x vocab
        return p_positions_mapped, p_target_vocab, h, c
