import numpy as np
import torch
import torch.nn as nn
import time
import torch.nn.functional as F
from torch.autograd import Variable


CUDA = torch.cuda.is_available()

# class CapsE(object):
#     def __init__(self, sequence_length, embedding_size, num_filters, vocab_size, iter_routing, batch_size=256,
#                  num_outputs_secondCaps=1, vec_len_secondCaps=10, initialization=[], filter_size=1, useConstantInit=False):
#         # Placeholders for input, output
#         self.input_x = tf.placeholder(tf.int32, [batch_size, sequence_length], name="input_x")
#         self.input_y = tf.placeholder(tf.float32, [batch_size, 1], name="input_y")
#         self.filter_size = filter_size
#         self.num_filters = num_filters
#         self.sequence_length = sequence_length
#         self.embedding_size = embedding_size
#         self.iter_routing = iter_routing
#         self.num_outputs_secondCaps = num_outputs_secondCaps
#         self.vec_len_secondCaps = vec_len_secondCaps
#         self.batch_size = batch_size
#         self.useConstantInit = useConstantInit
#         # Embedding layer
#         with tf.name_scope("embedding"):
#             if initialization == []:
#                 self.W = tf.Variable(
#                     tf.random_uniform([vocab_size, embedding_size], -math.sqrt(1.0 / embedding_size),
#                                       math.sqrt(1.0 / embedding_size), seed=1234), name="W")
#             else:
#                 self.W = tf.get_variable(name="W2", initializer=initialization)
#
#         self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
#         self.X = tf.expand_dims(self.embedded_chars, -1)
#
#         self.build_arch()
#         self.loss()
#         self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=500)
#
#         tf.logging.info('Seting up the main structure')
#
#     def build_arch(self):
#         #The first capsule layer
#         with tf.variable_scope('FirstCaps_layer'):
#             self.firstCaps = CapsLayer(num_outputs_secondCaps=self.num_outputs_secondCaps, vec_len_secondCaps=self.vec_len_secondCaps,
#                                     with_routing=False, layer_type='CONV', embedding_size=self.embedding_size,
#                                     batch_size=self.batch_size, iter_routing=self.iter_routing,
#                                     useConstantInit=self.useConstantInit, filter_size=self.filter_size,
#                                     num_filters=self.num_filters, sequence_length=self.sequence_length)
#
#             self.caps1 = self.firstCaps(self.X, kernel_size=1, stride=1)
#         #The second capsule layer
#         with tf.variable_scope('SecondCaps_layer'):
#             self.secondCaps = CapsLayer(num_outputs_secondCaps=self.num_outputs_secondCaps, vec_len_secondCaps=self.vec_len_secondCaps,
#                                     with_routing=True, layer_type='FC',
#                                     batch_size=self.batch_size, iter_routing=self.iter_routing,
#                                     embedding_size=self.embedding_size, useConstantInit=self.useConstantInit, filter_size=self.filter_size,
#                                     num_filters=self.num_filters, sequence_length=self.sequence_length)
#             self.caps2 = self.secondCaps(self.caps1)
#
#         self.v_length = tf.sqrt(tf.reduce_sum(tf.square(self.caps2), axis=2, keepdims=True) + epsilon)
#
#     def loss(self):
#         self.scores = tf.reshape(self.v_length, [self.batch_size, 1])
#         self.predictions = tf.nn.sigmoid(self.scores)
#         print("Using square softplus loss")
#         losses = tf.square(tf.nn.softplus(self.scores * self.input_y))
#         self.total_loss = tf.reduce_mean(losses)


class ConvKB(nn.Module):
    def __init__(self, input_dim, input_seq_len, in_channels, out_channels, drop_prob, alpha_leaky):
        super().__init__()

        self.conv_layer = nn.Conv2d(
            in_channels, out_channels, (1, input_seq_len))  # kernel size -> 1*input_seq_length(i.e. 2)
        self.dropout = nn.Dropout(drop_prob)
        self.non_linearity = nn.ReLU()
        self.fc_layer = nn.Linear((input_dim) * out_channels, 1)

        nn.init.xavier_uniform_(self.fc_layer.weight, gain=1.414)
        nn.init.xavier_uniform_(self.conv_layer.weight, gain=1.414)

    def forward(self, conv_input):

        batch_size, length, dim = conv_input.size()
        # assuming inputs are of the form ->
        conv_input = conv_input.transpose(1, 2)
        # batch * length(which is 3 here -> entity,relation,entity) * dim
        # To make tensor of size 4, where second dim is for input channels
        conv_input = conv_input.unsqueeze(1)

        out_conv = self.dropout(
            self.non_linearity(self.conv_layer(conv_input)))

        input_fc = out_conv.squeeze(-1).view(batch_size, -1)
        output = self.fc_layer(input_fc)
        return output


class SpecialSpmmFunctionFinal(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, edge, edge_w, N, E, out_features):
        # assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(
            edge, edge_w, torch.Size([N, N, out_features]))
        b = torch.sparse.sum(a, dim=1)
        ctx.N = b.shape[0]
        ctx.outfeat = b.shape[1]
        ctx.E = E
        ctx.indices = a._indices()[0, :]

        return b.to_dense()

    @staticmethod
    def backward(ctx, grad_output):
        grad_values = None
        if ctx.needs_input_grad[1]:
            edge_sources = ctx.indices

            if(CUDA):
                edge_sources = edge_sources.cuda()

            grad_values = grad_output[edge_sources]
            # grad_values = grad_values.view(ctx.E, ctx.outfeat)
            # print("Grad Outputs-> ", grad_output)
            # print("Grad values-> ", grad_values)
        return None, grad_values, None, None, None


class SpecialSpmmFinal(nn.Module):
    def forward(self, edge, edge_w, N, E, out_features):
        return SpecialSpmmFunctionFinal.apply(edge, edge_w, N, E, out_features)


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, num_nodes, in_features, out_features, nrela_dim, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_nodes = num_nodes
        self.alpha = alpha
        self.concat = concat
        self.nrela_dim = nrela_dim

        self.a = nn.Parameter(torch.zeros(
            size=(out_features, 2 * in_features + nrela_dim)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)
        self.a_2 = nn.Parameter(torch.zeros(size=(1, out_features)))
        nn.init.xavier_normal_(self.a_2.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm_final = SpecialSpmmFinal()

    def forward(self, input, edge, edge_embed, edge_list_nhop, edge_embed_nhop):
        N = input.size()[0]

        # Self-attention on the nodes - Shared attention mechanism
        edge = torch.cat((edge[:, :], edge_list_nhop[:, :]), dim=1)
        edge_embed = torch.cat(
            (edge_embed[:, :], edge_embed_nhop[:, :]), dim=0)

        edge_h = torch.cat(
            (input[edge[0, :], :], input[edge[1, :], :], edge_embed[:, :]), dim=1).t()
        # edge_h: (2*in_dim + nrela_dim) x E

        edge_m = self.a.mm(edge_h)
        # edge_m: D * E

        # to be checked later
        powers = -self.leakyrelu(self.a_2.mm(edge_m).squeeze())
        edge_e = torch.exp(powers).unsqueeze(1)
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm_final(
            edge, edge_e, N, edge_e.shape[0], 1)
        e_rowsum[e_rowsum == 0.0] = 1e-12

        e_rowsum = e_rowsum
        # e_rowsum: N x 1
        edge_e = edge_e.squeeze(1)

        edge_e = self.dropout(edge_e)
        # edge_e: E

        edge_w = (edge_e * edge_m).t()
        # edge_w: E * D

        h_prime = self.special_spmm_final(
            edge, edge_w, N, edge_w.shape[0], self.out_features)

        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out

        assert not torch.isnan(h_prime).any()
        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
