import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import math

def positional_encodings(T, D):
    encodings = torch.zeros(T, D)
    positions = torch.arange(0, T).float()
    for d in range(D):
        if d % 2 == 0:
            encodings[:, d] = torch.sin(positions / 10000 ** (d / D))
        else:
            encodings[:, d] = torch.cos(positions / 10000 ** (d / D))
    return encodings.unsqueeze(dim=0) # (1, T, D)

class Attention(nn.Module):
    def __init__(self, d_query, d_key, d_value):
        super(Attention, self).__init__()

        self.wq = nn.Linear(d_query, d_key)
        # self.wk = nn.Linear(d_key, d_key)
        # self.wv = nn.Linear(d_value, d_value)
        self.scale = math.sqrt(d_key)
        self.dropout = nn.Dropout()

    def forward(self, query, key, value):
        """
        query.shape:    (n, 1,      D)
        key.shape:      (n, p*2+1,  dim)
        value.shape:    (n, p*2+1,  1)
        return.shape:   (n, 1,      1)
        """
        query = self.wq(query)  # (n, 1, D)
        # key = self.wk(key)      # (n, p*2+1, dim)
        # value = self.wv(value)
        dot_products = torch.matmul(query, key.transpose(1, 2))
        return torch.matmul(self.dropout(F.softmax(dot_products / self.scale, dim=-1)), value)


class Shift(nn.Module):
    def __init__(self, args):
        super(Shift, self).__init__()

        self.p = 10 # max offset
        self.T = args.segment_length
        self.D = args.dim_feature
        self.use_attention = args.shift_with_attention
        self.group = [0, 1000, 2000]    # how features group up
        self.n_group = len(self.group) - 1

        if self.use_attention:
            # self.dim = 50
            # with torch.no_grad():
            #     self.positional_encodings = positional_encodings(self.p*2+1, self.dim).cuda()
            # self.attentions = nn.ModuleList([Attention(self.D, self.dim, 1) for i in range(self.D)])
            self.to_weights = [torch.randn(self.D, self.p*2+1).cuda() for i in range(self.n_group)]
        else:
            self.convs = nn.ModuleList([nn.Conv1d(self.group[i+1]-self.group[i], self.group[i+1]-self.group[i], 
                kernel_size=self.p*2+1, padding=self.p) for i in range(self.n_group)])

    def forward(self, x):
        """
        x.shape: (n, T, D)
        return.shape: (n, T, D)
        """
        if self.use_attention:
            with torch.no_grad():
                x_padded = torch.zeros(x.shape[0], self.T+self.p*2, self.D).cuda()
                x_padded[:,self.p:self.T+self.p,:] = x.clone() # (n, T+p*2, D)
            for i in range(self.n_group):
                to_weight = self.to_weights[i] # (D, p*2+1)
                weight = torch.matmul(x, to_weight) # (n, T, p*2+1)
                d = range(self.group[i], self.group[i+1])
                for t in range(self.T):
                    x[:,t,d] = torch.matmul(weight[:,t:t+1,:], x_padded[:,t:t+self.p*2+1,d]).squeeze(dim=1)
                # attention = self.attentions[d]
                # for t in range(self.T):
                #     x[:,t:t+1,d:d+1] = attention(
                #         x_padded[:,self.p+t:t+self.p+1,:],  # (n, 1,     D)
                #         self.positional_encodings,          # (n, p*2+1, D)
                #         x_padded[:,t:t+self.p*2+1,d:d+1]    # (n, p*2+1, 1)
                #     )
        else:
            for i in range(self.n_group):
                conv = self.convs[i]
                d = range(self.group[i], self.group[i+1])
                x[:,:,d] = conv(x[:,:,d].transpose(1, 2)).transpose(1, 2)
        return x

def logsumexp(x, dim=None):
    """
    Args:
        x: A pytorch tensor (any dimension will do)
        dim: int or None, over which to perform the summation. `None`, the
             default, performs over all axes.
    Returns: The result of the log(sum(exp(...))) operation.
    """
    if dim is None:
        xmax = x.max()
        xmax_ = x.max()
        return xmax_ + torch.log(torch.exp(x - xmax).sum())
    else:
        xmax, _ = x.max(dim, keepdim=True)
        xmax_, _ = x.max(dim)
        return xmax_ + torch.log(torch.exp(x - xmax).sum(dim))

class ChainCRF(nn.Module):
    def __init__(self, input_size, num_labels, bigram=True, **kwargs):
        '''
        Args:
            input_size: int
                the dimension of the input.
            num_labels: int
                the number of labels of the crf layer
            bigram: bool
                if apply bi-gram parameter.
            **kwargs:
        '''
        super(ChainCRF, self).__init__()
        self.input_size = input_size
        self.num_labels = num_labels + 1
        self.pad_label_id = num_labels
        self.bigram = bigram


        # state weight tensor
        self.state_nn = nn.Linear(input_size, self.num_labels)
        if bigram:
            # transition weight tensor
            self.trans_nn = nn.Linear(input_size, self.num_labels * self.num_labels)
            self.register_parameter('trans_matrix', None)
        else:
            self.trans_nn = None
            self.trans_matrix = Parameter(torch.Tensor(self.num_labels, self.num_labels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.state_nn.bias, 0.)
        if self.bigram:
            nn.init.xavier_uniform_(self.trans_nn.weight)
            nn.init.constant_(self.trans_nn.bias, 0.)
        else:
            nn.init.normal(self.trans_matrix)
        # if not self.bigram:
        #     nn.init.normal(self.trans_matrix)

    def forward(self, input, mask=None):
        '''
        Args:
            input: Tensor
                the input tensor with shape = [batch, length, input_size]
            mask: Tensor or None
                the mask tensor with shape = [batch, length]
        Returns: Tensor
            the energy tensor with shape = [batch, length, num_label, num_label]
        '''
        batch, length, _ = input.size()

        # compute out_s by tensor dot [batch, length, input_size] * [input_size, num_label]
        # thus out_s should be [batch, length, num_label] --> [batch, length, num_label, 1]
        out_s = self.state_nn(input).unsqueeze(2)

        if self.bigram:
            # compute out_s by tensor dot: [batch, length, input_size] * [input_size, num_label * num_label]
            # the output should be [batch, length, num_label,  num_label]
            out_t = self.trans_nn(input).view(batch, length, self.num_labels, self.num_labels)
            output = out_t + out_s
        else:
            # [batch, length, num_label, num_label]
            output = self.trans_matrix + out_s

        if mask is not None:
            output = output * mask.unsqueeze(2).unsqueeze(3)

        return output

    def loss(self, input, target, mask=None):
        '''
        Args:
            input: Tensor
                the input tensor with shape = [batch, length, input_size]
            target: Tensor
                the tensor of target labels with shape [batch, length]
            mask:Tensor or None
                the mask tensor with shape = [batch, length]
        Returns: Tensor
                A 1D tensor for minus log likelihood loss
        '''
        batch, length, _ = input.size()
        energy = self.forward(input, mask=mask)
        # shape = [length, batch, num_label, num_label]
        energy_transpose = energy.transpose(0, 1)
        # shape = [length, batch]
        target_transpose = target.transpose(0, 1)
        # shape = [length, batch, 1]
        mask_transpose = None
        if mask is not None:
            mask_transpose = mask.unsqueeze(2).transpose(0, 1)


        # shape = [batch, num_label]
        partition = None

        if input.is_cuda:
            # shape = [batch]
            batch_index = torch.arange(0, batch).long().cuda()
            prev_label = torch.cuda.LongTensor(batch).fill_(self.num_labels - 1)
            tgt_energy = Variable(torch.zeros(batch)).cuda()
        else:
            # shape = [batch]
            batch_index = torch.arange(0, batch).long()
            prev_label = torch.LongTensor(batch).fill_(self.num_labels - 1)
            tgt_energy = Variable(torch.zeros(batch))

        for t in range(length):
            # shape = [batch, num_label, num_label]
            curr_energy = energy_transpose[t]
            if t == 0:
                partition = curr_energy[:, -1, :]
            else:
                # shape = [batch, num_label]
                partition_new = logsumexp(curr_energy + partition.unsqueeze(2), dim=1)
                if mask_transpose is None:
                    partition = partition_new
                else:
                    mask_t = mask_transpose[t]
                    partition = partition + (partition_new - partition) * mask_t
            tgt_energy += curr_energy[batch_index, prev_label, target_transpose[t].data]
            prev_label = target_transpose[t].data

        return logsumexp(partition, dim=1) - tgt_energy

    def decode(self, input, mask=None, leading_symbolic=0):
        """
        Args:
            input: Tensor
                the input tensor with shape = [batch, length, input_size]
            mask: Tensor or None
                the mask tensor with shape = [batch, length]
            leading_symbolic: nt
                number of symbolic labels leading in type alphabets (set it to 0 if you are not sure)
        Returns: Tensor
            decoding results in shape [batch, length]
        """

        energy = self.forward(input, mask=mask).data

        # Input should be provided as (n_batch, n_time_steps, num_labels, num_labels)
        # For convenience, we need to dimshuffle to (n_time_steps, n_batch, num_labels, num_labels)
        energy_transpose = energy.transpose(0, 1)

        # the last row and column is the tag for pad symbol. reduce these two dimensions by 1 to remove that.
        # also remove the first #symbolic rows and columns.
        # now the shape of energies_shuffled is [n_time_steps, b_batch, t, t] where t = num_labels - #symbolic - 1.
        energy_transpose = energy_transpose[:, :, leading_symbolic:-1, leading_symbolic:-1]

        length, batch_size, num_label, _ = energy_transpose.size()

        if input.is_cuda:
            batch_index = torch.arange(0, batch_size).long().cuda()
            # pi = torch.zeros([length, batch_size, num_label, 1]).cuda()
            pi = torch.zeros([length, batch_size, num_label]).cuda()
            pointer = torch.cuda.LongTensor(length, batch_size, num_label).zero_()
            back_pointer = torch.cuda.LongTensor(length, batch_size).zero_()
        else:
            batch_index = torch.arange(0, batch_size).long()
            # pi = torch.zeros([length, batch_size, num_label, 1])
            pi = torch.zeros([length, batch_size, num_label])
            pointer = torch.LongTensor(length, batch_size, num_label).zero_()
            back_pointer = torch.LongTensor(length, batch_size).zero_()

        pi[0] = energy[:, 0, -1, leading_symbolic:-1]
        pointer[0] = -1
        for t in range(1, length):
            pi_prev = pi[t - 1].unsqueeze(dim=-1)
            # print(energy_transpose[t].shape)
            # print(pi_prev.shape)
            pi[t], pointer[t] = torch.max(energy_transpose[t] + pi_prev, dim=1)

        _, back_pointer[-1] = torch.max(pi[-1], dim=1)
        for t in reversed(range(length - 1)):
            pointer_last = pointer[t + 1]
            back_pointer[t] = pointer_last[batch_index, back_pointer[t + 1]]

        return back_pointer.transpose(0, 1) + leading_symbolic