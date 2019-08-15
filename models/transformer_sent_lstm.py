from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

from utils import maybe_cuda, setup_logger, unsort
from times_profiler import profiler

logger = setup_logger(__name__, 'train.log')
profilerLogger = setup_logger("profilerLogger", 'profiler.log', True)


def get_non_pad_mask(seq, lengths):
    #assert seq.dim() == 2
    batch, len_q = seq_k.size(0), seq_k.size(1)
    assert len(lengths) == batch
    return torch.tensor([[1 if i<clen else 0 for i in range(len_q) ] for clen in lengths]).type(torch.float).unsqueeze(-1)

def get_attn_key_pad_mask(seq_k, seq_q, lengths): # seq_k must be same as seq_q and (add CLS)
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    batch, len_q, len_k = seq_k.size(0), seq_q.size(1), seq_k.size(1)
    #padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = maybe_cuda(torch.ByteTensor([[0 if i<=clen else 1 for i in range(len_k) ] for clen in lengths]) )
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.out1 = nn.Linear(n_heads * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
    def forward(self, Q, K, V, attn_mask):
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) # attn_mask : [batch_size x n_heads x len_q x len_k]

        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context, attn = ScaledDotProductAttention(self.d_k)(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v) # context: [batch_size x len_q x n_heads * d_v]
        #print ("device", context.device, attn.device)
        output = self.out1(context)
        return self.layer_norm(output + residual), attn # output: [batch_size x len_q x d_model]

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_inner, dropout=0.1):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_inner, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_inner, out_channels=d_model, kernel_size=1)
        self.d_model = d_model
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs # inputs : [batch_size, len_q, d_model]
        output = self.relu(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_inner, dropout=dropout)

    def forward(self, enc_inputs, non_pad_mask=None, enc_self_attn_mask=None):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn

class SentenceEncoding(nn.Module):
    def __init__(self, len_max_seq, d_word_vec, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout=0.1):
        super(SentenceEncoding, self).__init__()
        self.CLS = maybe_cuda(torch.randn(1, 300))
        n_position = len_max_seq + 1
        self.d_model = d_model
        self.src_emb = nn.Linear(300, d_model)
        self.position_enc = nn.Embedding.from_pretrained( \
            get_sinusoid_encoding_table(n_position, d_word_vec), \
            freeze=True)
        self.layers = nn.ModuleList([ \
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout) for _ in range(n_layers)])
        self.dropout_func = nn.Dropout(dropout)
    def forward(self, enc_inputs, max_length, lengths):
        # (N, W, D)
        assert len(lengths) == len(enc_inputs)
        max_length += 1
        enc_inputs = self.add_cls_mark(enc_inputs)
        position = maybe_cuda(torch.tensor([[i if i<=clen else 0 for i in range(max_length) ] for clen in lengths]) )
        print ("--------->", type(enc_inputs), enc_inputs.size(), type(position), position.size())
        enc_outputs = self.src_emb(enc_inputs) + self.position_enc(position)
        #print ("--------->", type(enc_position), enc_position.size(), type(enc_outputs), enc_outputs.size())
        enc_self_attn_mask = get_attn_key_pad_mask(enc_inputs, enc_inputs, lengths)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask=enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        # enc_outputs : N W D
        x = enc_outputs[:, 0, :]
        #print ("--------->", type(enc_outputs), enc_outputs.size(), type(x), x.size())
        x = self.dropout_func(x)
        return x

    def add_cls_mark(self, enc_outputs):
        batch = enc_outputs.size(0)
        cls_mark = self.CLS.expand(batch, 1, self.CLS.size(1))
        enc_outputs = torch.cat((cls_mark, enc_outputs), 1)
        return enc_outputs


def zero_state(module, batch_size):
    # * 2 is for the two directions
    return Variable(maybe_cuda(torch.zeros(module.num_layers * 2, batch_size, module.hidden))), \
           Variable(maybe_cuda(torch.zeros(module.num_layers * 2, batch_size, module.hidden)))


class Model(nn.Module):
    def __init__(self, sentence_encoder, shidden=32*2, hidden=128, num_layers=2, len_max_seq=499):
        super(Model, self).__init__()
        self.len_max_seq = len_max_seq
        self.sentence_encoder = sentence_encoder

        self.sentence_lstm = nn.LSTM(input_size=shidden,
                                     hidden_size=hidden,
                                     num_layers=num_layers,
                                     batch_first=True,
                                     dropout=0,
                                     bidirectional=True)

        # We have two labels
        self.h2s = nn.Linear(hidden * 2, 2)

        self.num_layers = num_layers
        self.hidden = hidden

        self.criterion = nn.CrossEntropyLoss()


    def pad(self, s, max_length):
        s_length = s.size()[0]
        v = Variable(maybe_cuda(s.unsqueeze(0).unsqueeze(0)))
        padded = F.pad(v, (0, 0, 0, max_length - s_length))  # (1, 1, max_length, 300)
        shape = padded.size()
        return padded.view(shape[2], 1, shape[3])  # (max_length, 1, 300)


    def pad_document(self, d, max_document_length):
        d_length = d.size()[0]
        v = d.unsqueeze(0).unsqueeze(0)
        padded = F.pad(v, (0, 0,0, max_document_length - d_length ))  # (1, 1, max_length, 300)
        shape = padded.size()
        return padded.view(shape[2], 1, shape[3])  # (max_length, 1, 300)

    def forward(self, batch):
        batch_size = len(batch)

        sentences_per_doc = []
        all_batch_sentences = []
        for document in batch:
            all_batch_sentences.extend(document)
            sentences_per_doc.append(len(document))
        lengths = [s.size()[0] for s in all_batch_sentences]
        max_length = max(lengths) if max(lengths) < self.len_max_seq else self.len_max_seq-1
        padded_sentences = [self.pad(s, max_length) for s in all_batch_sentences]
        padded_sentences_tensor = torch.cat(padded_sentences, 1).permute(1, 0, 2)

        #print(type(padded_sentences_tensor), padded_sentences_tensor.size(), max_length, len(lengths))
        unsorted_encodings = self.sentence_encoder(padded_sentences_tensor, max_length, lengths)
        #print ("gdq -------------> ", len(all_batch_sentences), type(unsorted_encodings), unsorted_encodings.size())

        index = 0
        encoded_documents = []
        for sentences_count in sentences_per_doc:
            end_index = index + sentences_count
            encoded_documents.append(unsorted_encodings[index : end_index, :])
            index = end_index

        doc_sizes = [doc.size()[0] for doc in encoded_documents]
        max_doc_size = np.max(doc_sizes)
        ordered_document_idx = np.argsort(doc_sizes)[::-1]
        ordered_doc_sizes = sorted(doc_sizes)[::-1]
        ordered_documents = [encoded_documents[idx] for idx in ordered_document_idx]
        padded_docs = [self.pad_document(d, max_doc_size) for d in ordered_documents]
        docs_tensor = torch.cat(padded_docs, 1)
        packed_docs = pack_padded_sequence(docs_tensor, ordered_doc_sizes)
        sentence_lstm_output, _ = self.sentence_lstm(packed_docs, zero_state(self, batch_size=batch_size))
        padded_x, _ = pad_packed_sequence(sentence_lstm_output)  # (max sentence len, batch, 256)

        doc_outputs = []
        for i, doc_len in enumerate(ordered_doc_sizes):
            doc_outputs.append(padded_x[0:doc_len - 1, i, :])  # -1 to remove last prediction

        unsorted_doc_outputs = [doc_outputs[i] for i in unsort(ordered_document_idx)]
        sentence_outputs = torch.cat(unsorted_doc_outputs, 0)

        x = self.h2s(sentence_outputs)
        return x


def create():
    len_max_seq = 499
    d_word_vec = 512
    n_layers, n_head, d_k, d_v = 2, 8, 64, 64
    d_model = 512
    d_inner = 512
    dropout=0.5
    sentence_encoder = SentenceEncoding(len_max_seq, d_word_vec, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout)
    return Model(sentence_encoder, shidden=d_model, hidden = 256, num_layers=2, len_max_seq=len_max_seq)
