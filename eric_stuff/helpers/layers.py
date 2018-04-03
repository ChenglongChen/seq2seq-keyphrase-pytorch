import torch
import numpy as np
import torch.nn.functional as F
from helpers.embedding_helper import H5EmbeddingManager


def masked_softmax(x, m=None, axis=-1):
    '''
    Softmax with mask (optional)
    '''
    x = torch.clamp(x, min=-15.0, max=15.0)
    if m is not None:
        m = m.float()
        x = x * m
    e_x = torch.exp(x - torch.max(x, dim=axis, keepdim=True)[0])
    if m is not None:
        e_x = e_x * m
    softmax = e_x / (torch.sum(e_x, dim=axis, keepdim=True) + 1e-6)
    return softmax


def mask_distributions(p, mask):
    # mask and re-normalize
    res = p * mask
    _sum = torch.sum(res, dim=-1, keepdim=True)  # batch x time x 1
    epsilon = torch.le(_sum, 0.0).float() * 1e-6
    res = res / (_sum + epsilon)  # batch x time x vocab_size
    res = res * mask
    return res


class LayerNorm(torch.nn.Module):

    def __init__(self, input_dim):
        super(LayerNorm, self).__init__()
        self.gamma = torch.nn.Parameter(torch.ones(input_dim))
        self.beta = torch.nn.Parameter(torch.zeros(input_dim))
        self.eps = 1e-6

    def forward(self, x, mask):
        # x:        nbatch x hidden
        # mask:     nbatch
        mean = x.mean(-1, keepdim=True)
        std = torch.sqrt(x.var(dim=1, keepdim=True) + self.eps)
        output = self.gamma * (x - mean) / (std + self.eps) + self.beta
        return output * mask.unsqueeze(1)


class Embedding(torch.nn.Module):
    '''
    inputs: x:          batch x ...
    outputs:embedding:  batch x ... x emb
            mask:       batch x ...
    '''

    def __init__(self, embedding_size, vocab_size, trainable, id2word=None,
                 dropout_rate=0.0, embedding_oov_init='random',
                 embedding_type='random', pretrained_embedding_path=None,
                 vocab_oov_ext_size=0, enable_cuda=False):
        super(Embedding, self).__init__()
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.id2word = id2word
        self.embedding_type = embedding_type
        self.embedding_oov_init = embedding_oov_init
        self.pretrained_embedding_path = pretrained_embedding_path
        self.trainable = trainable
        self.enable_cuda = enable_cuda
        self.vocab_oov_ext_size = vocab_oov_ext_size
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.embedding_layer = torch.nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=0)
        self.init_weights()

    def init_weights(self):
        # Embeddings
        if self.embedding_type == 'random':
            word_embedding_init = np.random.uniform(low=-0.05, high=0.05, size=(self.vocab_size, self.embedding_size))
            word_embedding_init[0, :] = 0
        else:
            embedding_initr = H5EmbeddingManager(self.pretrained_embedding_path)
            word_embedding_init = embedding_initr.word_embedding_initialize(self.id2word,
                                                                            dim_size=self.embedding_size,
                                                                            mode=self.embedding_type,
                                                                            oov_init=self.embedding_oov_init)
            del embedding_initr
        # oov ext
        if self.vocab_oov_ext_size > 0:
            word_embedding_ext_init = np.random.uniform(low=-0.05, high=0.05, size=(self.vocab_oov_ext_size, self.embedding_size))
            word_embedding_init = np.concatenate([word_embedding_init, word_embedding_ext_init], axis=0)  # vocab+ext x emb_size

        word_embedding_init = torch.from_numpy(word_embedding_init).float()
        if self.enable_cuda:
            word_embedding_init = word_embedding_init.cuda()

        self.embedding_layer.weight = torch.nn.Parameter(word_embedding_init)
        if not self.trainable:
            self.embedding_layer.weight.requires_grad = False

    def compute_mask(self, x):
        mask = torch.ne(x, 0).float()
        return mask

    def embed(self, words):
        masked_embed_weight = self.embedding_layer.weight
        padding_idx = self.embedding_layer.padding_idx
        X = self.embedding_layer._backend.Embedding.apply(
            words, masked_embed_weight,
            padding_idx, self.embedding_layer.max_norm, self.embedding_layer.norm_type,
            self.embedding_layer.scale_grad_by_freq, self.embedding_layer.sparse)
        return X

    def forward(self, x):
        # drop entire word embeddings
        embeddings = self.embed(x)
        # apply standard dropout
        embeddings = self.dropout(embeddings)  # batch x time x emb
        mask = self.compute_mask(x)  # batch x time
        return embeddings, mask


class WeightDrop(torch.nn.Module):
    def __init__(self, module, weights, dropout=0, variational=False, enable_cuda=False):
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.variational = variational
        self._setup()

    def widget_demagnetizer_y2k_edition(*args, **kwargs):
        # We need to replace flatten_parameters with a nothing function
        # It must be a function rather than a lambda as otherwise pickling explodes
        # We can't write boring code though, so ... WIDGET DEMAGNETIZER Y2K EDITION!
        return

    def _setup(self):
        # Terrible temporary solution to an issue regarding compacting weights re: CUDNN RNN
        if issubclass(type(self.module), torch.nn.RNNBase):
            self.module.flatten_parameters = self.widget_demagnetizer_y2k_edition

        for name_w in self.weights:
            print('Applying weight drop of {} to {}'.format(self.dropout, name_w))
            w = getattr(self.module, name_w)
            del self.module._parameters[name_w]
            self.module.register_parameter(name_w + '_raw', torch.nn.Parameter(w.data))

    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            w = None
            if self.variational:
                mask = torch.autograd.Variable(torch.ones(raw_w.size(0), 1))
                if raw_w.is_cuda:
                    mask = mask.cuda()
                mask = torch.nn.functional.dropout(mask, p=self.dropout, training=True)
                w = mask.expand_as(raw_w) * raw_w
            else:
                w = torch.nn.functional.dropout(raw_w, p=self.dropout, training=self.training)
            setattr(self.module, name_w, w)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)


class LSTMCell(torch.nn.Module):

    """A basic LSTM cell."""

    def __init__(self, input_size, hidden_size, use_layernorm=False, use_bias=True):
        """
        Most parts are copied from torch.nn.LSTMCell.
        """

        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.use_layernorm = use_layernorm
        self.weight_ih = torch.nn.Parameter(torch.FloatTensor(input_size, 4 * hidden_size))
        self.weight_hh = torch.nn.Parameter(torch.FloatTensor(hidden_size, 4 * hidden_size))
        if use_bias:
            self.bias_f = torch.nn.Parameter(torch.FloatTensor(hidden_size))
            self.bias_iog = torch.nn.Parameter(torch.FloatTensor(3 * hidden_size))
        else:
            self.register_parameter('bias', None)
        if self.use_layernorm:
            self.layernorm_i = LayerNorm(input_dim=self.hidden_size * 4)
            self.layernorm_h = LayerNorm(input_dim=self.hidden_size * 4)
            self.layernorm_c = LayerNorm(input_dim=self.hidden_size)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.orthogonal(self.weight_hh.data)
        torch.nn.init.xavier_uniform(self.weight_ih.data, gain=1)
        if self.use_bias:
            self.bias_f.data.fill_(1.0)
            self.bias_iog.data.fill_(0.0)

    def get_init_hidden(self, bsz, use_cuda):
        h_0 = torch.autograd.Variable(torch.FloatTensor(bsz, self.hidden_size).zero_())
        c_0 = torch.autograd.Variable(torch.FloatTensor(bsz, self.hidden_size).zero_())

        if use_cuda:
            h_0, c_0 = h_0.cuda(), c_0.cuda()

        return h_0, c_0

    def forward(self, input_, mask_, h_0=None, c_0=None, dropped_h_0=None):
        """
        Args:
            input_:     A (batch, input_size) tensor containing input features.
            mask_:      (batch)
            hx:         A tuple (h_0, c_0), which contains the initial hidden
                        and cell state, where the size of both states is
                        (batch, hidden_size).
        Returns:
            h_1, c_1: Tensors containing the next hidden and cell state.
        """
        if h_0 is None or c_0 is None:
            h_init, c_init = self.get_init_hidden(input_.size(0), use_cuda=input_.is_cuda)
            if h_0 is None:
                h_0 = h_init

            if c_0 is None:
                c_0 = c_init

        if dropped_h_0 is None:
            dropped_h_0 = h_0

        # if (mask_.data == 0).all():
        #     return h_0, c_0
        wh = torch.mm(dropped_h_0, self.weight_hh)
        wi = torch.mm(input_, self.weight_ih)
        if self.use_layernorm:
            wi = self.layernorm_i(wi, mask_)
            wh = self.layernorm_h(wh, mask_)
        pre_act = wi + wh
        if self.use_bias:
            pre_act = pre_act + torch.cat([self.bias_f, self.bias_iog]).unsqueeze(0)

        f, i, o, g = torch.split(pre_act, split_size=self.hidden_size, dim=1)
        expand_mask_ = mask_.unsqueeze(1)  # batch x None
        c_1 = torch.sigmoid(f) * c_0 + torch.sigmoid(i) * torch.tanh(g)
        c_1 = c_1 * expand_mask_ + c_0 * (1 - expand_mask_)
        if self.use_layernorm:
            h_1 = torch.sigmoid(o) * torch.tanh(self.layernorm_c(c_1, mask_))
        else:
            h_1 = torch.sigmoid(o) * torch.tanh(c_1)
        h_1 = h_1 * expand_mask_ + h_0 * (1 - expand_mask_)
        return h_1, c_1

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class StackedLSTM(torch.nn.Module):
    '''
    inputs: x:          batch x time x emb
            mask:       batch x time
    outputs:
            encoding:   batch x time x h
            mask:       batch x time
    Dropout types:
        dropout_between_rnn_hiddens -- across time step
        dropout_between_rnn_layers -- if multi layer rnns
        dropout_in_rnn_weights -- rnn weight dropout
    '''

    def __init__(self, nemb, nhids, dropout_between_rnn_hiddens=0., dropout_in_rnn_weights=0., dropout_between_rnn_layers=0.,
                 use_layernorm=False, use_highway_connections=False, enable_cuda=False):
        super(StackedLSTM, self).__init__()
        self.nhids = nhids
        self.nemb = nemb
        self.dropout_in_rnn_weights = dropout_in_rnn_weights
        self.dropout_between_rnn_hiddens = dropout_between_rnn_hiddens
        self.dropout_between_rnn_layers = dropout_between_rnn_layers
        self.use_layernorm = use_layernorm
        self.use_highway_connections = use_highway_connections
        self.enable_cuda = enable_cuda
        self.nlayers = len(self.nhids)
        self.stack_rnns()
        if self.use_highway_connections:
            self.build_highway_connections()

    def build_highway_connections(self):
        highway_connections_x = [torch.nn.Linear(self.nemb if i == 0 else self.nhids[i - 1], self.nemb if i == 0 else self.nhids[i - 1]) for i in range(self.nlayers)]
        highway_connections_x_x = [torch.nn.Linear(self.nemb if i == 0 else self.nhids[i - 1], self.nhids[i], bias=False) for i in range(self.nlayers)]
        highway_connections_h = [torch.nn.Linear(self.nemb if i == 0 else self.nhids[i - 1], self.nhids[i]) for i in range(self.nlayers)]
        self.highway_connections_x = torch.nn.ModuleList(highway_connections_x)
        self.highway_connections_x_x = torch.nn.ModuleList(highway_connections_x_x)
        self.highway_connections_h = torch.nn.ModuleList(highway_connections_h)
        for i in range(self.nlayers):
            torch.nn.init.xavier_uniform(self.highway_connections_x[i].weight.data, gain=1)
            torch.nn.init.xavier_uniform(self.highway_connections_h[i].weight.data, gain=1)
            torch.nn.init.xavier_uniform(self.highway_connections_x_x[i].weight.data, gain=1)
            self.highway_connections_x[i].bias.data.fill_(0)
            self.highway_connections_h[i].bias.data.fill_(0)

    def stack_rnns(self):
        rnns = [LSTMCell(self.nemb if i == 0 else self.nhids[i - 1], self.nhids[i], use_layernorm=self.use_layernorm, use_bias=True)
                for i in range(self.nlayers)]
        if self.dropout_in_rnn_weights > 0.:
            print('Applying hidden weight dropout {:.2f}'.format(self.dropout_in_rnn_weights))
            rnns = [WeightDrop(rnn, ['weight_hh'], dropout=self.dropout_in_rnn_weights)
                    for rnn in rnns]
        self.rnns = torch.nn.ModuleList(rnns)

    def get_init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.enable_cuda:
            return [[(torch.autograd.Variable(weight.new(bsz, self.nhids[i]).zero_()).cuda(),
                     torch.autograd.Variable(weight.new(bsz, self.nhids[i]).zero_()).cuda())]
                    for i in range(self.nlayers)]
        else:
            return [[(torch.autograd.Variable(weight.new(bsz, self.nhids[i]).zero_()),
                     torch.autograd.Variable(weight.new(bsz, self.nhids[i]).zero_()))]
                    for i in range(self.nlayers)]

    def get_dropout_mask(self, x, _rate=0.5):
        mask = torch.ones(x.size())
        if self.training and _rate > 0.05:
            mask = mask.bernoulli_(1 - _rate) / (1 - _rate)
        mask = torch.autograd.Variable(mask, requires_grad=False)
        if self.enable_cuda:
            mask = mask.cuda()
        return mask

    def forward(self, x, mask, init_states=None):
        if init_states is None:
            state_stp = self.get_init_hidden(x.size(0))
        else:
            state_stp = init_states
        hidden_to_hidden_dropout_masks = [None for _ in range(self.nlayers)]

        for d, rnn in enumerate(self.rnns):
            for t in range(x.size(1)):

                input_mask = mask[:, t]
                if d == 0:
                    # 0th layer
                    curr_input = x[:, t]
                else:
                    curr_input = state_stp[d - 1][t][0]
                # apply dropout layer-to-layer
                drop_input = F.dropout(curr_input, p=self.dropout_between_rnn_layers, training=self.training) if d > 0 else curr_input
                previous_h, previous_c = state_stp[d][t]
                if t == 0:
                    # get hidden to hidden dropout mask at 0th time step of each rnn layer, and freeze them at teach time step
                    hidden_to_hidden_dropout_masks[d] = self.get_dropout_mask(previous_h, _rate=self.dropout_between_rnn_hiddens)
                dropped_previous_h = hidden_to_hidden_dropout_masks[d] * previous_h

                new_h, new_c = rnn.forward(drop_input, input_mask, previous_h, previous_c, dropped_previous_h)
                state_stp[d].append((new_h, new_c))

            if self.use_highway_connections:
                for t in range(x.size(1)):
                    input_mask = mask[:, t]
                    if d == 0:
                        # 0th layer
                        curr_input = x[:, t]
                    else:
                        curr_input = state_stp[d - 1][t][0]
                    new_h, new_c = state_stp[d][t]
                    gate_x = F.sigmoid(self.highway_connections_x[d].forward(curr_input))
                    gate_h = F.sigmoid(self.highway_connections_h[d].forward(curr_input))
                    new_h = self.highway_connections_x_x[d].forward(curr_input * gate_x) + gate_h * new_h  # batch x hid
                    new_h = new_h * input_mask.unsqueeze(1)
                    state_stp[d][t] = (new_h, new_c)

        states = [h[0] for h in state_stp[-1][1:]]  # list of batch x hid
        states = torch.stack(states, 1)  # batch x time x hid
        return states, mask


class UniLSTM(torch.nn.Module):
    '''
    inputs: x:          batch x time x emb
            mask:       batch x time
    outputs:encoding:   batch x time x hid
            last state: batch x hid
            mask:       batch x time
    Dropout types:
        dropouth -- dropout on hidden-to-hidden connections
        dropoutw -- hidden-to-hidden weight dropout
    '''

    def __init__(self, nemb, nhids, dropout_between_rnn_hiddens=0., dropout_in_rnn_weights=0., dropout_between_rnn_layers=0.,
                 use_layernorm=False, use_highway_connections=False, enable_cuda=False):
        super(UniLSTM, self).__init__()
        self.nhids = nhids
        self.nemb = nemb
        self.rnn = StackedLSTM(nemb=nemb,
                               nhids=nhids,
                               dropout_between_rnn_hiddens=dropout_between_rnn_hiddens,
                               dropout_between_rnn_layers=dropout_between_rnn_layers,
                               dropout_in_rnn_weights=dropout_in_rnn_weights,
                               use_layernorm=use_layernorm,
                               use_highway_connections=use_highway_connections,
                               enable_cuda=enable_cuda
                               )

    def forward(self, x, mask, init_states=None):
        # stacked rnn
        states, _ = self.rnn.forward(x, mask, init_states)
        last_state = states[:, -1]  # batch x hid
        states = states * mask.unsqueeze(-1)  # batch x time x hid
        return states, last_state, mask


class BiLSTM(torch.nn.Module):
    '''
    inputs: x:          batch x time x emb
            mask:       batch x time
    outputs:encoding:   batch x time x hid
            last state: batch x hid
            mask:       batch x time
    Dropout types:
        dropouth -- dropout on hidden-to-hidden connections
        dropoutw -- hidden-to-hidden weight dropout
    '''

    def __init__(self, nemb, nhids, dropout_between_rnn_hiddens=0., dropout_in_rnn_weights=0., dropout_between_rnn_layers=0.,
                 use_layernorm=False, use_highway_connections=False, enable_cuda=False):
        super(BiLSTM, self).__init__()
        self.nhids = nhids
        self.nemb = nemb
        self.enable_cuda = enable_cuda
        self.forward_rnn = StackedLSTM(nemb=self.nemb,
                                       nhids=[hid // 2 for hid in self.nhids],
                                       dropout_between_rnn_hiddens=dropout_between_rnn_hiddens,
                                       dropout_between_rnn_layers=dropout_between_rnn_layers,
                                       dropout_in_rnn_weights=dropout_in_rnn_weights,
                                       use_layernorm=use_layernorm,
                                       use_highway_connections=use_highway_connections,
                                       enable_cuda=enable_cuda
                                       )

        self.backward_rnn = StackedLSTM(nemb=self.nemb,
                                        nhids=[hid // 2 for hid in self.nhids],
                                        dropout_between_rnn_hiddens=dropout_between_rnn_hiddens,
                                        dropout_between_rnn_layers=dropout_between_rnn_layers,
                                        dropout_in_rnn_weights=dropout_in_rnn_weights,
                                        use_layernorm=use_layernorm,
                                        use_highway_connections=use_highway_connections,
                                        enable_cuda=enable_cuda
                                        )

    def flip(self, tensor, flip_dim=0):
        # flip
        idx = [i for i in range(tensor.size(flip_dim) - 1, -1, -1)]
        idx = torch.autograd.Variable(torch.LongTensor(idx))
        if self.enable_cuda:
            idx = idx.cuda()
        inverted_tensor = tensor.index_select(flip_dim, idx)
        return inverted_tensor

    def forward(self, x, mask):

        embeddings = x
        embeddings_inverted = self.flip(embeddings, flip_dim=1)  # batch x time x emb (backward)
        mask_inverted = self.flip(mask, flip_dim=1)  # batch x time (backward)

        forward_states, _ = self.forward_rnn.forward(embeddings, mask)  # batch x time x hid/2
        forward_last_state = forward_states[:, -1]  # batch x hid/2

        backward_states, _ = self.backward_rnn.forward(embeddings_inverted, mask_inverted)  # batch x time x hid/2 (backward)
        backward_last_state = backward_states[:, -1]  # batch x hid/2
        backward_states = self.flip(backward_states, flip_dim=1)  # batch x time x hid/2

        concat_states = torch.cat([forward_states, backward_states], -1)  # batch x time x hid
        concat_states = concat_states * mask.unsqueeze(-1)  # batch x time x hid
        concat_last_state = torch.cat([forward_last_state, backward_last_state], -1)  # batch x hid
        return concat_states, concat_last_state, mask


class TimeDistributedDense(torch.nn.Module):
    '''
    input:  x:          batch x time x a
            mask:       batch x time
    output: y:          batch x time x b
    '''

    def __init__(self, mlp):
        super(TimeDistributedDense, self).__init__()
        self.mlp = mlp

    def forward(self, x, mask):

        x_size = x.size()
        x = x.view(-1, x_size[-1])  # batch*time x a
        y = self.mlp.forward(x)  # batch*time x b
        y = y.view(x_size[:-1] + (y.size(-1),))  # batch x time x b
        y = y * mask.unsqueeze(-1)  # batch x time x b
        return y


class TimeDistributedRNN(torch.nn.Module):
    '''
    input:  embedding:  batch x T x time x emb
            mask:       batch x T x time
    output: sequence:   batch x T x time x enc
            last state: batch x T x enc
            mask:       batch x T x time
    '''

    def __init__(self, rnn):
        super(TimeDistributedRNN, self).__init__()
        self.rnn = rnn

    def forward(self, x, mask):

        x_size = x.size()
        x = x.view(-1, x_size[-2], x_size[-1])  # batch*T x time x emb
        _mask = mask.view(-1, mask.size(-1))  # batch*T x time
        seq, last, _ = self.rnn.forward(x, _mask)
        seq = seq.view(x_size[:-1] + (seq.size(-1),))  # batch x T x time x enc
        last = last.view(x_size[:-2] + (seq.size(-1),))  # batch x T x enc
        return seq, last, mask


class TimeDistributedEmbedding(torch.nn.Module):
    '''
    input:  embedding:  batch x T x time
    output: sequence:   batch x T x time x emb
            mask:       batch x T x time
    '''

    def __init__(self, emb_layer):
        super(TimeDistributedEmbedding, self).__init__()
        self.emb_layer = emb_layer

    def forward(self, x):
        x_size = x.size()
        x = x.view(-1, x_size[-1])  # batch*T x time
        emb, mask = self.emb_layer.forward(x)   # batch*T x time x emb
        emb = emb.view(x_size + (emb.size(-1),))
        mask = mask.view(x_size)
        return emb, mask


class FastBiLSTM(torch.nn.Module):
    """
    Adapted from https://github.com/facebookresearch/DrQA/
    now supports:   different rnn size for each layer
                    all zero rows in batch (from time distributed layer, by reshaping certain dimension)
    """

    def __init__(self, ninp, nhids, dropout_between_rnn_layers=0.):
        super(FastBiLSTM, self).__init__()
        self.ninp = ninp
        self.nhids = [h // 2 for h in nhids]
        self.nlayers = len(self.nhids)
        self.dropout_between_rnn_layers = dropout_between_rnn_layers
        self.stack_rnns()

    def stack_rnns(self):
        rnns = [torch.nn.LSTM(self.ninp if i == 0 else self.nhids[i - 1] * 2,
                              self.nhids[i],
                              num_layers=1,
                              bidirectional=True) for i in range(self.nlayers)]
        self.rnns = torch.nn.ModuleList(rnns)

    def forward(self, x, mask):

        def pad_(tensor, n):
            if n > 0:
                zero_pad = torch.autograd.Variable(torch.zeros((n,) + tensor.size()[1:]))
                if x.is_cuda:
                    zero_pad = zero_pad.cuda()
                tensor = torch.cat([tensor, zero_pad])
            return tensor

        """
        inputs: x:          batch x time x inp
                mask:       batch x time
        output: encoding:   batch x time x hidden[-1]
        """
        # Compute sorted sequence lengths
        batch_size = x.size(0)
        lengths = mask.data.eq(1).long().sum(1).squeeze()
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        lengths = list(lengths[idx_sort])
        idx_sort = torch.autograd.Variable(idx_sort)
        idx_unsort = torch.autograd.Variable(idx_unsort)

        # Sort x
        x = x.index_select(0, idx_sort)

        # remove non-zero rows, and remember how many zeros
        n_nonzero = np.count_nonzero(lengths)
        n_zero = batch_size - n_nonzero
        if n_zero != 0:
            lengths = lengths[:n_nonzero]
            x = x[:n_nonzero]

        # Transpose batch and sequence dims
        x = x.transpose(0, 1)

        # Pack it up
        rnn_input = torch.nn.utils.rnn.pack_padded_sequence(x, lengths)

        # Encode all layers
        outputs = [rnn_input]
        for i in range(self.nlayers):
            rnn_input = outputs[-1]

            # dropout between rnn layers
            if self.dropout_between_rnn_layers > 0:
                dropout_input = F.dropout(rnn_input.data,
                                          p=self.dropout_between_rnn_layers,
                                          training=self.training)
                rnn_input = torch.nn.utils.rnn.PackedSequence(dropout_input,
                                                              rnn_input.batch_sizes)
            seq, last = self.rnns[i](rnn_input)
            outputs.append(seq)
            if i == self.nlayers - 1:
                # last layer
                last_state = last[0]  # (num_layers * num_directions, batch, hidden_size)
                last_state = torch.cat([last_state[0], last_state[1]], 1)  # batch x hid_f+hid_b

        # Unpack everything
        for i, o in enumerate(outputs[1:], 1):
            outputs[i] = torch.nn.utils.rnn.pad_packed_sequence(o)[0]
        output = outputs[-1]

        # Transpose and unsort
        output = output.transpose(0, 1)  # batch x time x enc

        # re-padding
        output = pad_(output, n_zero)
        last_state = pad_(last_state, n_zero)

        output = output.index_select(0, idx_unsort)
        last_state = last_state.index_select(0, idx_unsort)

        # Pad up to original batch sequence length
        if output.size(1) != mask.size(1):
            padding = torch.zeros(output.size(0),
                                  mask.size(1) - output.size(1),
                                  output.size(2)).type(output.data.type())
            output = torch.cat([output, torch.autograd.Variable(padding)], 1)

        output = output.contiguous() * mask.unsqueeze(-1)
        return output, last_state, mask


class FastUniLSTM(torch.nn.Module):
    """
    Adapted from https://github.com/facebookresearch/DrQA/
    now supports:   different rnn size for each layer
                    all zero rows in batch (from time distributed layer, by reshaping certain dimension)
    """

    def __init__(self, ninp, nhids, dropout_between_rnn_layers=0.):
        super(FastUniLSTM, self).__init__()
        self.ninp = ninp
        self.nhids = nhids
        self.nlayers = len(self.nhids)
        self.dropout_between_rnn_layers = dropout_between_rnn_layers
        self.stack_rnns()

    def stack_rnns(self):
        rnns = [torch.nn.LSTM(self.ninp if i == 0 else self.nhids[i - 1],
                              self.nhids[i],
                              num_layers=1,
                              bidirectional=False) for i in range(self.nlayers)]
        self.rnns = torch.nn.ModuleList(rnns)

    def forward(self, x, mask):

        def pad_(tensor, n):
            if n > 0:
                zero_pad = torch.autograd.Variable(torch.zeros((n,) + tensor.size()[1:]))
                if x.is_cuda:
                    zero_pad = zero_pad.cuda()
                tensor = torch.cat([tensor, zero_pad])
            return tensor

        """
        inputs: x:          batch x time x inp
                mask:       batch x time
        output: encoding:   batch x time x hidden[-1]
        """
        # Compute sorted sequence lengths
        batch_size = x.size(0)
        lengths = mask.data.eq(1).long().sum(1).squeeze()
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        lengths = list(lengths[idx_sort])
        idx_sort = torch.autograd.Variable(idx_sort)
        idx_unsort = torch.autograd.Variable(idx_unsort)

        # Sort x
        x = x.index_select(0, idx_sort)

        # remove non-zero rows, and remember how many zeros
        n_nonzero = np.count_nonzero(lengths)
        n_zero = batch_size - n_nonzero
        if n_zero != 0:
            lengths = lengths[:n_nonzero]
            x = x[:n_nonzero]

        # Transpose batch and sequence dims
        x = x.transpose(0, 1)

        # Pack it up
        rnn_input = torch.nn.utils.rnn.pack_padded_sequence(x, lengths)

        # Encode all layers
        outputs = [rnn_input]
        for i in range(self.nlayers):
            rnn_input = outputs[-1]

            # dropout between rnn layers
            if self.dropout_between_rnn_layers > 0:
                dropout_input = F.dropout(rnn_input.data,
                                          p=self.dropout_between_rnn_layers,
                                          training=self.training)
                rnn_input = torch.nn.utils.rnn.PackedSequence(dropout_input,
                                                              rnn_input.batch_sizes)
            seq, last = self.rnns[i](rnn_input)
            outputs.append(seq)
            if i == self.nlayers - 1:
                # last layer
                last_state = last[0]  # (num_layers * num_directions, batch, hidden_size)
                last_state = last_state[0]  # batch x hidden_size

        # Unpack everything
        for i, o in enumerate(outputs[1:], 1):
            outputs[i] = torch.nn.utils.rnn.pad_packed_sequence(o)[0]
        output = outputs[-1]

        # Transpose and unsort
        output = output.transpose(0, 1)  # batch x time x enc

        # re-padding
        output = pad_(output, n_zero)
        last_state = pad_(last_state, n_zero)

        output = output.index_select(0, idx_unsort)
        last_state = last_state.index_select(0, idx_unsort)

        # Pad up to original batch sequence length
        if output.size(1) != mask.size(1):
            padding = torch.zeros(output.size(0),
                                  mask.size(1) - output.size(1),
                                  output.size(2)).type(output.data.type())
            output = torch.cat([output, torch.autograd.Variable(padding)], 1)

        output = output.contiguous() * mask.unsqueeze(-1)
        return output, last_state, mask


class VAEDecoder(torch.nn.Module):
    '''
    input: embeddings:  batch x time x emb
           masks:       batch x time
           mu:          batch x z_dim
           logvar:      batch x zdim
    output:
    '''
    def __init__(self, input_dim, z_dim, nhids, ntokens,
                 z_trans_input_dim=0, z_trans_rnn_hidden_size=[],
                 dropout_between_rnn_hiddens=0., dropout_in_rnn_weights=0., dropout_between_rnn_layers=0.,
                 use_layernorm=False, fast_rnns=False, enable_cuda=False):
        super(VAEDecoder, self).__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.nhids = nhids
        self.ntokens = ntokens
        self.dropout_between_rnn_layers = dropout_between_rnn_layers
        self.dropout_in_rnn_weights = dropout_in_rnn_weights
        self.dropout_between_rnn_hiddens = dropout_between_rnn_hiddens
        self.use_layernorm = use_layernorm
        self.z_trans_rnn_hidden_size = z_trans_rnn_hidden_size
        self.z_trans_input_dim = z_trans_input_dim
        self.use_z_trans = len(z_trans_rnn_hidden_size) > 0
        self.enable_cuda = enable_cuda

        if fast_rnns:
            self.rnns = FastUniLSTM(ninp=self.input_dim,
                                    nhids=self.nhids,
                                    dropout_between_rnn_layers=self.dropout_between_rnn_layers)
        else:
            self.rnns = UniLSTM(nemb=self.input_dim, nhids=self.nhids,
                                dropout_between_rnn_hiddens=self.dropout_between_rnn_hiddens,
                                dropout_between_rnn_layers=self.dropout_between_rnn_layers,
                                dropout_in_rnn_weights=self.dropout_in_rnn_weights,
                                use_layernorm=self.use_layernorm,
                                use_highway_connections=False,
                                enable_cuda=self.enable_cuda)
        self.z_theta = TimeDistributedDense(torch.nn.Linear(self.nhids[-1] + self.z_dim, self.input_dim))
        self.decoder = TimeDistributedDense(torch.nn.Linear(self.input_dim, self.ntokens))
        if self.use_z_trans:
            if fast_rnns:
                self.z_trans_rnns = FastUniLSTM(ninp=self.z_trans_input_dim,
                                                nhids=self.z_trans_rnn_hidden_size)
            else:
                self.z_trans_rnns = UniLSTM(nemb=self.z_trans_input_dim, nhids=self.z_trans_rnn_hidden_size,
                                            enable_cuda=self.enable_cuda)
        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform(self.z_theta.mlp.weight.data)  # glorot init
        torch.nn.init.xavier_uniform(self.decoder.mlp.weight.data)  # glorot init
        self.z_theta.mlp.bias.data.fill_(0)
        self.decoder.mlp.bias.data.fill_(0)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.autograd.Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)

    def forward(self, x, x_mask, mu, logvar):

        z = self.reparameterize(mu, logvar)  # batch x z_dim
        if self.use_z_trans:
            z = z.view(z.size(0), -1, self.z_trans_input_dim)  # batch x z_time x z_input_dim
            if self.enable_cuda:
                z_mask = torch.autograd.Variable(torch.ones(z.size(0), z.size(1)).cuda())  # fake mask
            else:
                z_mask = torch.autograd.Variable(torch.ones(z.size(0), z.size(1)))  # fake mask
            _, z_trans_last_states, _ = self.z_trans_rnns.forward(z, z_mask)  # batch x z_time x z_hid
            z = z_trans_last_states.view(z_trans_last_states.size(0), -1)  # batch x z_time*z_hid

        states, _, _ = self.rnns.forward(x, x_mask)  # batch x time x hid
        z = torch.stack([z] * states.size(1), dim=1)  # batch x time x z_dim
        states = torch.cat([states, z], 2)  # batch x time x hid+z_dim
        output = F.tanh(self.z_theta.forward(states, x_mask))  # batch x time x inp_dim
        output = output * x_mask.unsqueeze(2)  # batch x time x hid+z_dim
        logits = self.decoder.forward(output, x_mask)  # batch x time x ntokens
        pred = masked_softmax(logits, m=x_mask.view(x_mask.size(0), x_mask.size(1), 1), axis=-1)
        return pred


class MLPDecoder(torch.nn.Module):
    '''
    input:  x:  batch x input_dim
    output: y:  batch x output_dim (default 1)
    '''

    def __init__(self, input_dim, hidden_dim, output_dim=1, activation='relu', last_activation='None', enable_cuda=False):
        super(MLPDecoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.enable_cuda = enable_cuda
        self.activation = activation
        self.last_activation = last_activation
        self.nlayers = len(hidden_dim)
        mlps = [torch.nn.Linear(self.input_dim if i == 0 else self.hidden_dim[i - 1], self.hidden_dim[i]) for i in range(self.nlayers)]
        self.mlps = torch.nn.ModuleList(mlps)
        self.last_layer = torch.nn.Linear(self.hidden_dim[-1], output_dim, bias=False)
        self.init_weights()

    def init_weights(self):
        for i in range(self.nlayers):
            torch.nn.init.xavier_uniform(self.mlps[i].weight.data, gain=1)
            self.mlps[i].bias.data.fill_(0)
        torch.nn.init.xavier_uniform(self.last_layer.weight.data, gain=1)

    def forward(self, x):
        curr = x
        for i in range(self.nlayers):
            curr = self.mlps[i].forward(curr)
            if self.activation == 'relu':
                curr = F.relu(curr)
            elif self.activation == 'tanh':
                curr = F.tanh(curr)
            elif self.activation == 'sigmoid':
                curr = F.sigmoid(curr)
            else:
                raise NotImplemented
        curr = self.last_layer.forward(curr)
        if self.last_activation == 'softmax':
            curr = masked_softmax(curr, axis=-1)
        else:
            pass
        return curr


class BoundaryDecoderAttention(torch.nn.Module):
    '''
        input:  p:          batch x inp_p
                p_mask:     batch
                q:          batch x time x inp_q
                q_mask:     batch x time
                h_tm1:      batch x out
                depth:      int
        output: z:          batch x inp_p+inp_q
    '''

    def __init__(self, input_dim, output_dim, enable_cuda=False):
        super(BoundaryDecoderAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.enable_cuda = enable_cuda

        self.V = torch.nn.Linear(self.input_dim, self.output_dim)
        self.W_a = torch.nn.Linear(self.output_dim, self.output_dim)
        self.v = torch.nn.Parameter(torch.FloatTensor(self.output_dim))
        self.c = torch.nn.Parameter(torch.FloatTensor(1))
        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform(self.V.weight.data, gain=1)
        torch.nn.init.xavier_uniform(self.W_a.weight.data, gain=1)
        self.V.bias.data.fill_(0)
        self.W_a.bias.data.fill_(0)
        torch.nn.init.normal(self.v.data, mean=0, std=0.05)
        self.c.data.fill_(1.0)

    def forward(self, H_r, mask_r, h_tm1):
        # H_r: batch x time x inp
        # mask_r: batch x time
        # h_tm1: batch x out
        batch_size, time = H_r.size(0), H_r.size(1)
        Fk = self.V.forward(H_r.view(-1, H_r.size(2)))  # batch*time x out
        Fk_prime = self.W_a.forward(h_tm1)  # batch x out
        Fk = Fk.view(batch_size, time, -1)  # batch x time x out
        Fk = torch.tanh(Fk + Fk_prime.unsqueeze(1))  # batch x time x out

        beta = torch.matmul(Fk, self.v)  # batch x time
        beta = beta + self.c.unsqueeze(0)  # batch x time
        beta = masked_softmax(beta, mask_r, axis=-1)  # batch x time
        z = torch.bmm(beta.view(beta.size(0), 1, beta.size(1)), H_r)  # batch x 1 x inp
        z = z.view(z.size(0), -1)  # batch x inp
        return z, beta


class BoundaryDecoder(torch.nn.Module):
    '''
    input:  observation:    batch x time x inp_dim
            ob mask:        batch x time
            history:        batch x history_dim
    output: y:  batch x time x 2
    '''

    def __init__(self, input_dim, hidden_dim, enable_cuda=False):
        super(BoundaryDecoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.enable_cuda = enable_cuda
        self.attention_layer = BoundaryDecoderAttention(input_dim=input_dim,
                                                        output_dim=hidden_dim,
                                                        enable_cuda=enable_cuda)

        self.rnn = LSTMCell(self.input_dim, self.hidden_dim, use_layernorm=False, use_bias=True)

    def forward(self, x, x_mask, h_0):

        state_stp = [(h_0, h_0)]
        beta_list = []
        if self.enable_cuda:
            mask = torch.autograd.Variable(torch.ones(x.size(0)).cuda())  # fake mask
        else:
            mask = torch.autograd.Variable(torch.ones(x.size(0)))  # fake mask
        for t in range(2):

            previous_h, previous_c = state_stp[t]
            curr_input, beta = self.attention_layer.forward(x, x_mask, h_tm1=previous_h)
            new_h, new_c = self.rnn(curr_input, mask, previous_h, previous_c, previous_h)
            state_stp.append((new_h, new_c))
            beta_list.append(beta)

        # beta list: list of batch x time
        res = torch.stack(beta_list, 2)  # batch x time x 2
        res = res * x_mask.unsqueeze(2)  # batch x time x 2
        return res


class LSTMDecoder(torch.nn.Module):
    '''
    input: embeddings:  batch x time x emb
           masks:       batch x time
           init_states: batch x hidden_dim
    output:
    '''
    def __init__(self, input_dim, hidden_dim, ntokens,
                 dropout_between_rnn_hiddens=0., dropout_in_rnn_weights=0., dropout_between_rnn_layers=0.,
                 use_layernorm=False, fast_rnns=False, enable_cuda=False):
        super(LSTMDecoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.ntokens = ntokens
        self.dropout_between_rnn_layers = dropout_between_rnn_layers
        self.dropout_in_rnn_weights = dropout_in_rnn_weights
        self.dropout_between_rnn_hiddens = dropout_between_rnn_hiddens
        self.use_layernorm = use_layernorm
        self.enable_cuda = enable_cuda

        self.rnns = UniLSTM(nemb=self.input_dim, nhids=[hidden_dim],
                            dropout_between_rnn_hiddens=self.dropout_between_rnn_hiddens,
                            dropout_between_rnn_layers=self.dropout_between_rnn_layers,
                            dropout_in_rnn_weights=self.dropout_in_rnn_weights,
                            use_layernorm=self.use_layernorm,
                            use_highway_connections=False,
                            enable_cuda=self.enable_cuda)
        self.decoder = TimeDistributedDense(torch.nn.Linear(self.hidden_dim, self.ntokens))
        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform(self.decoder.mlp.weight.data)  # glorot init
        self.decoder.mlp.bias.data.fill_(0)

    def forward(self, x, x_mask, init_states):

        states, _, _ = self.rnns.forward(x, x_mask, [[(init_states, init_states)]])  # batch x time x hid
        states = states * x_mask.unsqueeze(2)  # batch x time x hid
        logits = self.decoder.forward(states, x_mask)  # batch x time x ntokens
        pred = masked_softmax(logits, m=x_mask.view(x_mask.size() + (1,)), axis=-1)
        return pred


class MultiMLPDecoder(torch.nn.Module):
    '''
    input:  [x1: batch x input_1_dim
            x2: batch x input_2_dim
            x3: batch x input_3_dim
            ...
            xn: batch x input_n_dim]
    output: y:  batch x output_dim (default 1)
    '''

    def __init__(self, input_dim, hidden_dim, output_dim=1, enable_cuda=False):
        super(MultiMLPDecoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.enable_cuda = enable_cuda
        self.nlayers = len(hidden_dim)
        first_layer = [torch.nn.Linear(self.input_dim[i], self.hidden_dim[0]) for i in range(len(input_dim))]
        self.first_layer = torch.nn.ModuleList(first_layer)
        if self.nlayers > 1:
            mlps = [torch.nn.Linear(self.hidden_dim[i - 1], self.hidden_dim[i]) for i in range(1, self.nlayers)]
            self.mlps = torch.nn.ModuleList(mlps)
        self.last_layer = torch.nn.Linear(self.hidden_dim[-1], output_dim, bias=False)
        self.init_weights()

    def init_weights(self):
        for i in range(len(self.input_dim)):
            torch.nn.init.xavier_uniform(self.first_layer[i].weight.data, gain=1)
            self.first_layer[i].bias.data.fill_(0)
        for i in range(self.nlayers - 1):
            torch.nn.init.xavier_uniform(self.mlps[i].weight.data, gain=1)
            self.mlps[i].bias.data.fill_(0)
        torch.nn.init.xavier_uniform(self.last_layer.weight.data, gain=1)

    def forward(self, x):
        transfered = []
        for i, item in enumerate(x):
            temp = self.first_layer[i].forward(item)
            temp = F.tanh(temp)
            transfered.append(temp)
        transfered = torch.stack(transfered, -1)
        transfered = torch.mean(transfered, -1)  # batch x hidden[0]
        curr = transfered
        for i in range(self.nlayers - 1):
            curr = self.mlps[i].forward(curr)
            curr = F.tanh(curr)
        curr = self.last_layer.forward(curr)
        return curr


class AttentionLSTMCell(torch.nn.Module):

    """A basic LSTM cell."""

    def __init__(self, input_size, doc_enc_size, hidden_size, attn_size, use_layernorm=False, use_bias=True):
        """
        Most parts are copied from torch.nn.LSTMCell.
        """

        super(AttentionLSTMCell, self).__init__()
        self.input_size = input_size
        self.doc_enc_size = doc_enc_size
        self.hidden_size = hidden_size
        self.attn_size = attn_size
        self.use_bias = use_bias
        self.use_layernorm = use_layernorm

        self.lstm_cell_0 = LSTMCell(self.input_size, self.hidden_size, use_layernorm=self.use_layernorm, use_bias=self.use_bias)
        self.lstm_cell_1 = LSTMCell(self.doc_enc_size, self.hidden_size, use_layernorm=self.use_layernorm, use_bias=self.use_bias)

        self.h_half_to_attn = torch.nn.Linear(self.hidden_size, self.attn_size)
        self.attn_dim_reduce = TimeDistributedDense(torch.nn.Linear(self.attn_size, 1))
        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform(self.h_half_to_attn.weight.data, gain=1)
        torch.nn.init.xavier_uniform(self.attn_dim_reduce.mlp.weight.data, gain=1)
        self.h_half_to_attn.bias.data.fill_(0)
        self.attn_dim_reduce.mlp.bias.data.fill_(0)

    def forward(self, input_, mask_,
                source_encoding_sequence, source_encoding_sequence_attn_term, source_mask,
                list_of_condition_encoding_attn_term,
                h_0=None, c_0=None, dropped_h_0=None):
        """
        Args:
            input_:                             (batch, input_size) tensor containing input features.
            mask_:                              (batch, )
            source_encoding_sequence:           (batch, time, source_enc)
            source_encoding_sequence_attn_term: (batch, time, attn_dim)
            source_mask:                        (batch, time)
            condition_encoding_attn_term:       list of (batch, time, attn_dim)
            h_0:                                (batch, hidden_size), initial hidden state.
            c_0:                                (batch, hidden_size), initial cell state.
            dropped_h_0:                        (batch, hidden_size), dropped initial hidden state.
        Returns:
            h_1, c_1:                           (batch, hidden_size), Tensors containing the next hidden and cell state.
            weighted_source_enc:                (batch, source_enc)
            attn:                               (batch, time)
        """
        h_half, c_half = self.lstm_cell_0.forward(input_, mask_, h_0, c_0, dropped_h_0)

        h_half_attn_term = self.h_half_to_attn.forward(h_half)  # batch x attn_dim
        h_half_attn_term = torch.stack([h_half_attn_term] * source_encoding_sequence.size(1), dim=1)  # batch x time x attn_dim
        alpha = h_half_attn_term + source_encoding_sequence_attn_term  # batch x time x attn_dim
        for condition_encoding_attn_term in list_of_condition_encoding_attn_term:
            alpha = alpha + condition_encoding_attn_term  # batch x time x attn_dim
        alpha = torch.tanh(alpha)  # batch x time x attn_dim
        alpha = self.attn_dim_reduce.forward(alpha, source_mask).squeeze(2)  # batch x time
        alpha = masked_softmax(alpha, source_mask, axis=-1)  # batch x time

        weighted_source_enc = source_encoding_sequence * alpha.unsqueeze(-1)  # batch x time x source_enc
        weighted_source_enc = torch.sum(weighted_source_enc, 1)  # batch x source_enc

        h_1, c_1 = self.lstm_cell_1.forward(weighted_source_enc, mask_, h_half, c_half, h_half)
        return h_1, c_1, weighted_source_enc, alpha


class GenerativeAttnDecoder(torch.nn.Module):
    '''
    input: embeddings:                  batch x time x emb
           masks:                       batch x time
           init_states_h:               batch x hidden_dim
           init_states_c:               batch x hidden_dim
           source_encoding_sequence:    batch x source_time x source_enc_dim
           condition_encoding:          batch x condition_enc_dim
           condition2_encoding:         batch x condition2_enc_dim

    output:
    '''
    def __init__(self, input_dim, hidden_dim, source_enc_dim, cond_enc_dim, attn_dim,
                 dropout_between_rnn_hiddens=0., dropout_input=0.,
                 use_layernorm=False, enable_cuda=False):
        super(GenerativeAttnDecoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.source_enc_dim = source_enc_dim
        if not isinstance(cond_enc_dim, list):
            cond_enc_dim = [cond_enc_dim]
        self.cond_enc_dim = cond_enc_dim
        self.attn_dim = attn_dim
        self.dropout_input = dropout_input
        self.dropout_between_rnn_hiddens = dropout_between_rnn_hiddens
        self.use_layernorm = use_layernorm
        self.enable_cuda = enable_cuda
        self.source_enc_to_attn = TimeDistributedDense(torch.nn.Linear(self.source_enc_dim, self.attn_dim))
        cond_enc_to_attn = [torch.nn.Linear(self.cond_enc_dim[i], self.attn_dim) for i in range(len(cond_enc_dim))]
        self.cond_enc_to_attn = torch.nn.ModuleList(cond_enc_to_attn)
        self.rnn_cell = AttentionLSTMCell(input_size=input_dim, doc_enc_size=source_enc_dim, hidden_size=hidden_dim, attn_size=attn_dim,
                                          use_layernorm=self.use_layernorm)

        self.init_weights()

    def init_weights(self):

        torch.nn.init.xavier_uniform(self.source_enc_to_attn.mlp.weight.data)
        self.source_enc_to_attn.mlp.bias.data.fill_(0)
        for i in range(len(self.cond_enc_dim)):
            torch.nn.init.xavier_uniform(self.cond_enc_to_attn[i].weight.data)
            self.cond_enc_to_attn[i].bias.data.fill_(0)

    def get_init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.enable_cuda:
            return [(torch.autograd.Variable(weight.new(bsz, self.hidden_dim).zero_()).cuda(),
                    torch.autograd.Variable(weight.new(bsz, self.hidden_dim).zero_()).cuda())]
        else:
            return [(torch.autograd.Variable(weight.new(bsz, self.hidden_dim).zero_()),
                    torch.autograd.Variable(weight.new(bsz, self.hidden_dim).zero_()))]

    def get_dropout_mask(self, x, _rate=0.5):
        mask = torch.ones(x.size())
        if self.training and _rate > 0.05:
            mask = mask.bernoulli_(1 - _rate) / (1 - _rate)
        mask = torch.autograd.Variable(mask, requires_grad=False)
        if self.enable_cuda:
            mask = mask.cuda()
        return mask

    def forward(self, target_embedding, target_mask, init_states_h, init_states_c, source_encoding_sequence, source_mask, condition_encoding):
        assert isinstance(condition_encoding, list) and len(condition_encoding) == len(self.cond_enc_dim)
        source_encoding_sequence_attn_term = self.source_enc_to_attn.forward(source_encoding_sequence, source_mask)  # batch x source_time x attn_dim
        list_of_condition_encoding_attn_term = []
        for i in range(len(self.cond_enc_dim)):
            condition_encoding_attn_term = self.cond_enc_to_attn[i].forward(condition_encoding[i])  # batch x attn_dim
            condition_encoding_attn_term = torch.stack([condition_encoding_attn_term] * source_encoding_sequence.size(1), 1)  # batch x source_time x attn_dim
            list_of_condition_encoding_attn_term.append(condition_encoding_attn_term)

        if init_states_h is None:
            state_stp = self.get_init_hidden(target_embedding.size(0))
        else:
            state_stp = [(init_states_h, init_states_c)]
        hidden_to_hidden_dropout_masks = None
        weighted_source_enc_list, alpha_list = [], []

        for t in range(target_embedding.size(1)):

            input_ = target_embedding[:, t]
            mask_ = target_mask[:, t]
            # apply dropout layer-to-layer
            drop_input = F.dropout(input_, p=self.dropout_input, training=self.training)
            previous_h, previous_c = state_stp[t]
            if t == 0:
                # get hidden to hidden dropout mask at 0th time step, and freeze them at teach time step
                hidden_to_hidden_dropout_masks = self.get_dropout_mask(previous_h, _rate=self.dropout_between_rnn_hiddens)
            dropped_previous_h = hidden_to_hidden_dropout_masks * previous_h

            new_h, new_c, weighted_source_enc, alpha = self.rnn_cell.forward(drop_input, mask_,
                                                                             source_encoding_sequence, source_encoding_sequence_attn_term, source_mask,
                                                                             list_of_condition_encoding_attn_term,
                                                                             h_0=previous_h, c_0=previous_c, dropped_h_0=dropped_previous_h)
            weighted_source_enc_list.append(weighted_source_enc)
            alpha_list.append(alpha)
            state_stp.append((new_h, new_c))

        hidden_states_h = torch.stack([s[0] for s in state_stp[1:]], 1)  # batch x time x hid
        hidden_states_c = torch.stack([s[1] for s in state_stp[1:]], 1)  # batch x time x hid
        weighted_source_encodings = torch.stack(weighted_source_enc_list, 1)  # batch x time x enc_hid
        attentions = torch.stack(alpha_list, 1)  # batch x time x source_time
        hidden_states_h_last = hidden_states_h[:, -1]  # batch x hid
        hidden_states_c_last = hidden_states_c[:, -1]  # batch x hid
        # mask
        hidden_states_h = hidden_states_h * target_mask.unsqueeze(-1)
        hidden_states_c = hidden_states_c * target_mask.unsqueeze(-1)
        weighted_source_encodings = weighted_source_encodings * target_mask.unsqueeze(-1)
        attentions = attentions * target_mask.unsqueeze(-1)
        return hidden_states_h, hidden_states_c, hidden_states_h_last, hidden_states_c_last, weighted_source_encodings, attentions


class GenerateFromVocabulary(torch.nn.Module):
    '''
    input: target embeddings:               batch x time x emb
           target masks:                    batch x time
           target decodings:                batch x time x tar_dec
           weighted sum source encoding:    batch x time x source_enc
           condition encoding:              batch x cond_enc

    output:
    '''
    def __init__(self, emb_dim, dec_dim, hidden_dim, source_enc_dim, cond_enc_dim, ntokens):
        super(GenerateFromVocabulary, self).__init__()
        self.emb_dim = emb_dim
        self.dec_dim = dec_dim
        self.hidden_dim = hidden_dim
        self.source_enc_dim = source_enc_dim
        if not isinstance(cond_enc_dim, list):
            cond_enc_dim = [cond_enc_dim]
        self.cond_enc_dim = cond_enc_dim
        self.ntokens = ntokens

        self.target_emb_to_hid = TimeDistributedDense(torch.nn.Linear(self.emb_dim, self.hidden_dim))
        self.target_dec_to_hid = TimeDistributedDense(torch.nn.Linear(self.dec_dim, self.hidden_dim))
        self.source_enc_to_hid = TimeDistributedDense(torch.nn.Linear(self.source_enc_dim, self.hidden_dim))
        cond_enc_to_hid = [torch.nn.Linear(self.cond_enc_dim[i], self.hidden_dim) for i in range(len(cond_enc_dim))]
        self.cond_enc_to_hid = torch.nn.ModuleList(cond_enc_to_hid)
        self.decoder = TimeDistributedDense(torch.nn.Linear(self.hidden_dim, self.ntokens))

        self.init_weights()

    def init_weights(self):

        torch.nn.init.xavier_uniform(self.target_emb_to_hid.mlp.weight.data)
        self.target_emb_to_hid.mlp.bias.data.fill_(0)
        torch.nn.init.xavier_uniform(self.target_dec_to_hid.mlp.weight.data)
        self.target_dec_to_hid.mlp.bias.data.fill_(0)
        torch.nn.init.xavier_uniform(self.source_enc_to_hid.mlp.weight.data)
        self.source_enc_to_hid.mlp.bias.data.fill_(0)
        for i in range(len(self.cond_enc_dim)):
            torch.nn.init.xavier_uniform(self.cond_enc_to_hid[i].weight.data)
            self.cond_enc_to_hid[i].bias.data.fill_(0)
        torch.nn.init.xavier_uniform(self.decoder.mlp.weight.data)
        self.decoder.mlp.bias.data.fill_(0)

    def forward(self, target_embedding, target_mask, target_decoding, weighted_sum_source_encoding, condition_encoding):
        assert isinstance(condition_encoding, list) and len(condition_encoding) == len(self.cond_enc_dim)
        res = self.target_emb_to_hid.forward(target_embedding, target_mask)  # batch x time x hid
        res = res + self.target_dec_to_hid.forward(target_decoding, target_mask)  # batch x time x hid
        res = res + self.source_enc_to_hid.forward(weighted_sum_source_encoding, target_mask)  # batch x time x hid
        for i in range(len(self.cond_enc_dim)):
            temp = self.cond_enc_to_hid[i].forward(condition_encoding[i])  # batch x hid
            res = res + torch.stack([temp] * res.size(1), dim=1)  # batch x time x hid
        res = res * target_mask.unsqueeze(-1)  # batch x time x hid
        res = torch.tanh(res)
        res = self.decoder.forward(res, target_mask)  # batch x time x vocab
        res = masked_softmax(res, target_mask.unsqueeze(-1), axis=-1)  # batch x time x vocab

        return res


class PointerSoftmax(torch.nn.Module):
    '''
    input:  target decodings:               batch x time x tar_dec
            target masks:                   batch x time
            weighted sum source encoding:   batch x time x source_enc
            probs over source positions:    batch x time x source_time
            probs over target vocab:        batch x time x vocab_size
    output:
            weighted probs distribution over:
            source positions:               batch x time x source_time
            target vocab:                   batch x time x vocab_size
            the weights:                    batch x time
    '''
    def __init__(self, dec_dim, source_enc_dim, hidden_dim):
        super(PointerSoftmax, self).__init__()
        self.dec_dim = dec_dim
        self.hidden_dim = hidden_dim
        self.source_enc_dim = source_enc_dim

        self.target_dec_to_hid = TimeDistributedDense(torch.nn.Linear(self.dec_dim, self.hidden_dim))
        self.source_enc_to_hid = TimeDistributedDense(torch.nn.Linear(self.source_enc_dim, self.hidden_dim))
        self.square_layer = TimeDistributedDense(torch.nn.Linear(self.hidden_dim, self.hidden_dim))
        self.squash_layer = TimeDistributedDense(torch.nn.Linear(self.hidden_dim, 1))

        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform(self.target_dec_to_hid.mlp.weight.data)
        self.target_dec_to_hid.mlp.bias.data.fill_(0)
        torch.nn.init.xavier_uniform(self.source_enc_to_hid.mlp.weight.data)
        self.source_enc_to_hid.mlp.bias.data.fill_(0)
        torch.nn.init.xavier_uniform(self.square_layer.mlp.weight.data)
        self.square_layer.mlp.bias.data.fill_(0)
        torch.nn.init.xavier_uniform(self.squash_layer.mlp.weight.data)
        self.squash_layer.mlp.bias.data.fill_(0)

    def forward(self, target_decoding, target_mask, weighted_sum_source_encoding, p_source_position, p_target_vocab):
        proj1 = self.target_dec_to_hid.forward(target_decoding, target_mask) +\
            self.source_enc_to_hid.forward(weighted_sum_source_encoding, target_mask)
        proj1 = torch.tanh(proj1) * target_mask.unsqueeze(-1)  # batch x time x hid
        proj2 = self.square_layer.forward(proj1, target_mask)  # batch x time x hid
        proj2 = torch.tanh(proj2) * target_mask.unsqueeze(-1)  # batch x time x hid
        proj3 = self.squash_layer.forward(proj1 + proj2, target_mask).squeeze(-1)  # batch x time
        switch = torch.sigmoid(proj3).unsqueeze(-1)  # batch x time x 1
        return p_source_position * switch * target_mask.unsqueeze(-1),\
            p_target_vocab * (1.0 - switch) * target_mask.unsqueeze(-1),\
            switch.squeeze(-1)


class MergeDistributions(torch.nn.Module):
    '''
    input:  probs over source positions:        batch x time x source_time
            probs over target vocab:            batch x time x vocab_size
            source:                             batch x source_time

    output:
            probs over target vocab (mapped):   batch x time x vocab_size
    '''
    def __init__(self, enable_cuda=False):
        super(MergeDistributions, self).__init__()
        self.enable_cuda = enable_cuda

    def to_one_hot(self, y, vocab_size):
        y_tensor = y.data if isinstance(y, torch.autograd.Variable) else y
        y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
        y_one_hot = torch.zeros(y_tensor.size()[0], vocab_size).scatter_(1, y_tensor, 1)
        y_one_hot = y_one_hot.view(*y.shape, -1)
        res = torch.autograd.Variable(y_one_hot) if isinstance(y, torch.autograd.Variable) else y_one_hot
        if self.enable_cuda:
            res = res.cuda()
        return res

    def forward(self, p_source_position, p_target_vocab, input_source):
        vocab_size = p_target_vocab.size(-1)
        source_one_hot = self.to_one_hot(input_source, vocab_size)  # batch x source_time x vocab_size
        vocab_probs = torch.bmm(p_source_position, source_one_hot)  # batch x time x vocab_size
        return vocab_probs  # batch x time x vocab_size


class MaskDistributions(torch.nn.Module):
    '''
    input:  probs over target vocab:            batch x time x vocab_size
            mask over target vocab:             batch x time x vocab_size

    output:
            probs over target vocab (masked):   batch x time x vocab_size
    '''
    def __init__(self):
        super(MaskDistributions, self).__init__()

    def forward(self, p_vocab, mask):
        res = p_vocab * mask
        tmp_sum = torch.sum(p_vocab, dim=-1, keepdim=True)
        _sum = torch.sum(res, dim=-1, keepdim=True)  # batch x time x 1
        epsilon = torch.le(_sum, 0.0).float() * 1e-6  # batch x time x 1
        res = res / (_sum + epsilon)  # batch x time x vocab_size
        res = res * tmp_sum  # batch x time x vocab_size
        res = res * mask
        return res


class MultiAttentionLSTMCell(torch.nn.Module):

    """Attention between two LSTM cells."""

    def __init__(self, input_size, enc_size, hidden_size, attn_size, use_layernorm=False, use_bias=True):
        """
        Most parts are copied from torch.nn.LSTMCell.
        """

        super(MultiAttentionLSTMCell, self).__init__()
        self.input_size = input_size
        self.enc_size = enc_size
        self.hidden_size = hidden_size
        self.attn_size = attn_size
        self.use_bias = use_bias
        self.use_layernorm = use_layernorm

        self.lstm_cell_0 = LSTMCell(self.input_size, self.hidden_size, use_layernorm=self.use_layernorm, use_bias=self.use_bias)
        self.lstm_cell_1 = LSTMCell(self.enc_size, self.hidden_size, use_layernorm=self.use_layernorm, use_bias=self.use_bias)

        self.h_half_to_attn = torch.nn.Linear(self.hidden_size, self.attn_size)
        self.attn_dim_reduce = TimeDistributedDense(torch.nn.Linear(self.attn_size, 1))
        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform(self.h_half_to_attn.weight.data, gain=1)
        torch.nn.init.xavier_uniform(self.attn_dim_reduce.mlp.weight.data, gain=1)
        self.h_half_to_attn.bias.data.fill_(0)
        self.attn_dim_reduce.mlp.bias.data.fill_(0)

    def forward(self, input_, mask_,
                condition_encoding_sequence, condition_encoding_sequence_attn_term, condition_mask,
                prev_game_info,
                h_0=None, c_0=None, dropped_h_0=None):
        """
        Args:
            input_:                                 (batch, input_size) tensor containing input features.
            mask_:                                  (batch, )
            condition_encoding_sequence:            list of (batch, time_i, enc)
            condition_encoding_sequence_attn_term:  list of (batch, time_i, attn_dim)
            condition_mask:                         list of (batch, time_i)
            prev_game_info:                         batch x hidden
            h_0:                                    (batch, hidden_size), initial hidden state.
            c_0:                                    (batch, hidden_size), initial cell state.
            dropped_h_0:                            (batch, hidden_size), dropped initial hidden state.
        Returns:
            h_1, c_1:                               (batch, hidden_size), Tensors containing the next hidden and cell state.
            weighted_source_enc:                    list of (batch, enc)
            attn:                                   list of (batch, time_i)
        """
        h_half, c_half = self.lstm_cell_0.forward(input_, mask_, h_0, c_0, dropped_h_0)
        h_half_attn_term = self.h_half_to_attn.forward(h_half + prev_game_info)  # batch x attn_dim
        if not isinstance(condition_encoding_sequence, list):
            condition_encoding_sequence = [condition_encoding_sequence]
        if not isinstance(condition_encoding_sequence_attn_term, list):
            condition_encoding_sequence_attn_term = [condition_encoding_sequence_attn_term]
        if not isinstance(condition_mask, list):
            condition_mask = [condition_mask]

        list_of_alpha = []
        list_of_weighted_condition_enc = []
        input_to_2nd_lstm = None
        for i in range(len(condition_encoding_sequence)):
            temp_h_halp_attn_term = torch.stack([h_half_attn_term] * condition_encoding_sequence[i].size(1), dim=1)  # batch x time x attn_dim
            alpha = temp_h_halp_attn_term + condition_encoding_sequence_attn_term[i]  # batch x time x attn_dim
            alpha = torch.tanh(alpha)  # batch x time x attn_dim
            alpha = self.attn_dim_reduce.forward(alpha, condition_mask[i]).squeeze(2)  # batch x time
            alpha = masked_softmax(alpha, condition_mask[i], axis=-1)  # batch x time
            list_of_alpha.append(alpha)

            weighted_cond_enc = condition_encoding_sequence[i] * alpha.unsqueeze(-1)  # batch x time x enc
            weighted_cond_enc = torch.sum(weighted_cond_enc, 1)  # batch x enc
            list_of_weighted_condition_enc.append(weighted_cond_enc)

        input_to_2nd_lstm = torch.sum(torch.stack(list_of_weighted_condition_enc, -1), -1)  # batch  x enc
        h_1, c_1 = self.lstm_cell_1.forward(input_to_2nd_lstm, mask_, h_half, c_half, h_half)
        return h_1, c_1, list_of_weighted_condition_enc, list_of_alpha


class GenerativeMultiAttnDecoder(torch.nn.Module):
    '''
    input: embeddings:                      batch x time x emb
           masks:                           batch x time
           init_states_h:                   batch x hidden_dim
           init_states_c:                   batch x hidden_dim
           condition_encoding_sequence:     list of batch x time_i x enc_dim
           condition_mask:                  list of batch x time_i
           history_info:                    info from previous generation batch x info_hid

    output:
           hidden_states_h:                 batch x time x dec
           hidden_states_c:                 batch x time x dec
           weighted_condition_encodings:    list of batch x time x enc
           attentions:                      list of batch x time x time_i
    '''
    def __init__(self, input_dim, hidden_dim, enc_dim, attn_dim, prev_info_dim, condition_amount=1,
                 dropout_between_rnn_hiddens=0., dropout_input=0.,
                 use_layernorm=False, enable_cuda=False):
        super(GenerativeMultiAttnDecoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.enc_dim = enc_dim
        self.attn_dim = attn_dim
        self.condition_amount = condition_amount
        self.dropout_input = dropout_input
        self.dropout_between_rnn_hiddens = dropout_between_rnn_hiddens
        self.use_layernorm = use_layernorm
        self.enable_cuda = enable_cuda
        enc_to_attn = [TimeDistributedDense(torch.nn.Linear(self.enc_dim, self.attn_dim)) for _ in range(condition_amount)]
        self.enc_to_attn = torch.nn.ModuleList(enc_to_attn)
        self.history_to_hidden = torch.nn.Linear(prev_info_dim, self.hidden_dim)
        self.rnn_cell = MultiAttentionLSTMCell(input_size=input_dim, enc_size=enc_dim, hidden_size=hidden_dim, attn_size=attn_dim,
                                               use_layernorm=self.use_layernorm)

        self.init_weights()

    def init_weights(self):
        for i in range(self.condition_amount):
            torch.nn.init.xavier_uniform(self.enc_to_attn[i].mlp.weight.data)
            self.enc_to_attn[i].mlp.bias.data.fill_(0)
        torch.nn.init.xavier_uniform(self.history_to_hidden.weight.data)
        self.history_to_hidden.bias.data.fill_(0)

    def get_init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.enable_cuda:
            return [(torch.autograd.Variable(weight.new(bsz, self.hidden_dim).zero_()).cuda(),
                    torch.autograd.Variable(weight.new(bsz, self.hidden_dim).zero_()).cuda())]
        else:
            return [(torch.autograd.Variable(weight.new(bsz, self.hidden_dim).zero_()),
                    torch.autograd.Variable(weight.new(bsz, self.hidden_dim).zero_()))]

    def get_dropout_mask(self, x, _rate=0.5):
        mask = torch.ones(x.size())
        if self.training and _rate > 0.05:
            mask = mask.bernoulli_(1 - _rate) / (1 - _rate)
        mask = torch.autograd.Variable(mask, requires_grad=False)
        if self.enable_cuda:
            mask = mask.cuda()
        return mask

    def forward(self, target_embedding, target_mask, init_states_h, init_states_c, condition_encoding_sequence, condition_mask, history_info):

        if not isinstance(condition_encoding_sequence, list):
            condition_encoding_sequence = [condition_encoding_sequence]
        if not isinstance(condition_mask, list):
            condition_mask = [condition_mask]

        list_of_condition_encoding_attn_term = []
        for i in range(len(condition_encoding_sequence)):
            encoding_sequence_attn_term = self.enc_to_attn[i].forward(condition_encoding_sequence[i], condition_mask[i])  # batch x cond_time x attn_dim
            list_of_condition_encoding_attn_term.append(encoding_sequence_attn_term)

        history_info = self.history_to_hidden.forward(history_info)  # batch x hidden

        if init_states_h is None:
            state_stp = self.get_init_hidden(target_embedding.size(0))
        else:
            state_stp = [(init_states_h, init_states_c)]
        hidden_to_hidden_dropout_masks = None
        weighted_condition_enc_list, alpha_list = [], []

        for t in range(target_embedding.size(1)):

            input_ = target_embedding[:, t]
            mask_ = target_mask[:, t]
            # apply dropout layer-to-layer
            drop_input = F.dropout(input_, p=self.dropout_input, training=self.training)
            previous_h, previous_c = state_stp[t]
            if t == 0:
                # get hidden to hidden dropout mask at 0th time step, and freeze them at each time step
                hidden_to_hidden_dropout_masks = self.get_dropout_mask(previous_h, _rate=self.dropout_between_rnn_hiddens)
            dropped_previous_h = hidden_to_hidden_dropout_masks * previous_h

            new_h, new_c, list_of_weighted_enc, list_of_alpha = self.rnn_cell.forward(drop_input, mask_,
                                                                                      condition_encoding_sequence, list_of_condition_encoding_attn_term, condition_mask,
                                                                                      history_info,
                                                                                      h_0=previous_h, c_0=previous_c, dropped_h_0=dropped_previous_h)
            weighted_condition_enc_list.append(list_of_weighted_enc)
            alpha_list.append(list_of_alpha)
            state_stp.append((new_h, new_c))

        hidden_states_h = torch.stack([s[0] for s in state_stp[1:]], 1)  # batch x time x hid
        hidden_states_c = torch.stack([s[1] for s in state_stp[1:]], 1)  # batch x time x hid
        weighted_condition_encodings, attentions = [], []
        for i in range(len(condition_encoding_sequence)):
            weighted_condition_encodings.append(torch.stack([enc[i] for enc in weighted_condition_enc_list], 1))  # batch x time x enc_hid
            attentions.append(torch.stack([a[i] for a in alpha_list], 1))  # batch x time x time_i

        # mask
        hidden_states_h = hidden_states_h * target_mask.unsqueeze(-1)
        hidden_states_c = hidden_states_c * target_mask.unsqueeze(-1)

        weighted_condition_encodings = [item * target_mask.unsqueeze(-1) for item in weighted_condition_encodings]
        attentions = [item * target_mask.unsqueeze(-1) for item in attentions]
        return hidden_states_h, hidden_states_c, weighted_condition_encodings, attentions


class MultiGenerateFromVocabulary(torch.nn.Module):
    '''
    input: target embeddings:               batch x time x emb
           target masks:                    batch x time
           target decodings:                batch x time x tar_dec
           weighted sum condition encoding: batch x time x enc
    output:
           prob distribution over vocab:    batch x time x vocab
    '''
    def __init__(self, emb_dim, dec_dim, hidden_dim, enc_dim, ntokens, condition_amount=1,):
        super(MultiGenerateFromVocabulary, self).__init__()
        self.emb_dim = emb_dim
        self.dec_dim = dec_dim
        self.hidden_dim = hidden_dim
        self.enc_dim = enc_dim
        self.ntokens = ntokens
        self.target_emb_to_hid = TimeDistributedDense(torch.nn.Linear(self.emb_dim, self.hidden_dim))
        self.target_dec_to_hid = TimeDistributedDense(torch.nn.Linear(self.dec_dim, self.hidden_dim))
        self.condition_amount = condition_amount
        enc_to_hid = [TimeDistributedDense(torch.nn.Linear(self.enc_dim, self.hidden_dim)) for _ in range(condition_amount)]
        self.enc_to_hid = torch.nn.ModuleList(enc_to_hid)

        self.decoder = TimeDistributedDense(torch.nn.Linear(self.hidden_dim, self.ntokens))

        self.init_weights()

    def init_weights(self):

        torch.nn.init.xavier_uniform(self.target_emb_to_hid.mlp.weight.data)
        self.target_emb_to_hid.mlp.bias.data.fill_(0)
        torch.nn.init.xavier_uniform(self.target_dec_to_hid.mlp.weight.data)
        self.target_dec_to_hid.mlp.bias.data.fill_(0)
        for i in range(self.condition_amount):
            torch.nn.init.xavier_uniform(self.enc_to_hid[i].mlp.weight.data)
            self.enc_to_hid[i].mlp.bias.data.fill_(0)
        torch.nn.init.xavier_uniform(self.decoder.mlp.weight.data)
        self.decoder.mlp.bias.data.fill_(0)

    def forward(self, target_embedding, target_mask, target_decoding, weighted_sum_condition_encoding):
        if not isinstance(weighted_sum_condition_encoding, list):
            weighted_sum_condition_encoding = [weighted_sum_condition_encoding]

        res = self.target_emb_to_hid.forward(target_embedding, target_mask)  # batch x time x hid
        res = res + self.target_dec_to_hid.forward(target_decoding, target_mask)  # batch x time x hid
        for i in range(len(weighted_sum_condition_encoding)):
            res = res + self.enc_to_hid[i].forward(weighted_sum_condition_encoding[i], target_mask)  # batch x hid
        res = res * target_mask.unsqueeze(-1)  # batch x time x hid
        res = torch.tanh(res)
        res = self.decoder.forward(res, target_mask)  # batch x time x vocab
        res = masked_softmax(res, target_mask.unsqueeze(-1), axis=-1)  # batch x time x vocab

        return res


class MultiPointerSoftmax(torch.nn.Module):
    '''
    input:  target decodings:                   batch x time x tar_dec
            target masks:                       batch x time
            weighted sum condition encoding:    list of batch x time x enc
            probs over condition positions:     list of batch x time x time_i
            probs over target vocab:            batch x time x vocab_size
    output:
            weighted probs distribution over:
            condition positions:                list of batch x time x time_i
            target vocab:                       batch x time x vocab_size
            the weights:                        batch x time x condition_amount+1
    '''
    def __init__(self, dec_dim, enc_dim, hidden_dim, condition_amount=1):
        super(MultiPointerSoftmax, self).__init__()
        self.dec_dim = dec_dim
        self.hidden_dim = hidden_dim
        self.enc_dim = enc_dim
        self.target_dec_to_hid = TimeDistributedDense(torch.nn.Linear(self.dec_dim, self.hidden_dim))
        self.condition_amount = condition_amount
        enc_to_hid = [TimeDistributedDense(torch.nn.Linear(self.enc_dim, self.hidden_dim)) for _ in range(condition_amount)]
        self.enc_to_hid = torch.nn.ModuleList(enc_to_hid)

        self.square_layer = TimeDistributedDense(torch.nn.Linear(self.hidden_dim, self.hidden_dim))
        self.squash_layer = TimeDistributedDense(torch.nn.Linear(self.hidden_dim, condition_amount + 1))

        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform(self.target_dec_to_hid.mlp.weight.data)
        self.target_dec_to_hid.mlp.bias.data.fill_(0)
        for i in range(self.condition_amount):
            torch.nn.init.xavier_uniform(self.enc_to_hid[i].mlp.weight.data)
            self.enc_to_hid[i].mlp.bias.data.fill_(0)
        torch.nn.init.xavier_uniform(self.square_layer.mlp.weight.data)
        self.square_layer.mlp.bias.data.fill_(0)
        torch.nn.init.xavier_uniform(self.squash_layer.mlp.weight.data)
        self.squash_layer.mlp.bias.data.fill_(0)

    def forward(self, target_decoding, target_mask, weighted_sum_condition_encoding, p_condition_position, p_target_vocab):
        if not isinstance(weighted_sum_condition_encoding, list):
            weighted_sum_condition_encoding = [weighted_sum_condition_encoding]

        proj1 = self.target_dec_to_hid.forward(target_decoding, target_mask)
        for i in range(len(weighted_sum_condition_encoding)):
            proj1 = proj1 + self.enc_to_hid[i].forward(weighted_sum_condition_encoding[i], target_mask)
        proj1 = torch.tanh(proj1) * target_mask.unsqueeze(-1)  # batch x time x hid

        proj2 = self.square_layer.forward(proj1, target_mask)  # batch x time x hid
        proj2 = torch.tanh(proj2) * target_mask.unsqueeze(-1)  # batch x time x hid

        proj3 = self.squash_layer.forward(proj1 + proj2, target_mask)  # batch x time x condition_amount+1
        switch = masked_softmax(proj3, axis=-1)  # batch x time x cond_amount+1
        res = []
        for i, p in enumerate(p_condition_position + [p_target_vocab]):
            res.append(p * switch[:, :, i].unsqueeze(-1) * target_mask.unsqueeze(-1))  # batch x time x vocab_i

        return res[:-1], res[-1], switch
