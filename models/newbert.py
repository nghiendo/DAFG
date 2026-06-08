import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel

from layers import DynamicLSTM, GraphConvolution


ROBERTA_PATH = './roberta'


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias


def masked_average(hidden_states, mask):
    mask = mask.float().unsqueeze(-1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (hidden_states * mask).sum(dim=1) / denom


def normalize_adjacency(adj, mask):
    mask_2d = mask.float().unsqueeze(1) * mask.float().unsqueeze(2)
    adj = adj * mask_2d
    eye = torch.eye(adj.size(-1), device=adj.device).unsqueeze(0)
    adj = adj + eye * mask.float().unsqueeze(2)
    degree = adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return adj / degree


class RelationAwareSelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, aspect_hidden, feature_hidden, relation_bias, src_mask):
        batch_size, seq_len, hidden_dim = feature_hidden.size()
        query = self.q_proj(aspect_hidden).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(feature_hidden).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(feature_hidden).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + relation_bias.unsqueeze(1)

        key_mask = src_mask.unsqueeze(1).unsqueeze(2).bool()
        scores = scores.masked_fill(~key_mask, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, value)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        return self.out_proj(context)


class DAGFEncoder(nn.Module):
    def __init__(self, mlm_model, opt):
        super().__init__()
        self.opt = opt
        self.mlm_model = mlm_model
        self.roberta_model = RobertaModel.from_pretrained(ROBERTA_PATH)

        self.mlm_hidden = nn.Linear(opt.roberta_dim, opt.bert_dim)
        self.layer_norm = LayerNorm(opt.bert_dim)
        self.dropout = nn.Dropout(opt.bert_dropout)

        self.syn_gcn1 = GraphConvolution(opt.bert_dim, opt.bert_dim)
        self.syn_gcn2 = GraphConvolution(opt.bert_dim, opt.bert_dim)
        self.sem_gcn = GraphConvolution(opt.bert_dim, opt.bert_dim)

        self.syn_conv = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=(5, 5), padding=(2, 2)),
            nn.ReLU(),
        )
        self.syn_conv_proj = nn.Linear(10, opt.bert_dim)
        self.syn_lstm = DynamicLSTM(opt.bert_dim, opt.hidden_dim // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.syn_lstm_proj = nn.Linear(opt.hidden_dim, opt.bert_dim)
        self.syn_attention = RelationAwareSelfAttention(opt.bert_dim, opt.attention_heads, opt.dropout)

        self.kl_syn_proj = nn.Linear(opt.bert_dim, opt.bert_dim)
        self.kl_sem_proj = nn.Linear(opt.bert_dim, opt.bert_dim)

    def _build_syn_representation(self, mlm_hidden, aspect_hidden, syn_adj, relation_bias, src_mask):
        syn_hidden = F.relu(self.syn_gcn1(mlm_hidden, syn_adj))
        syn_hidden = self.dropout(F.relu(self.syn_gcn2(syn_hidden, syn_adj)))

        cnn_features = self.syn_conv(syn_adj.unsqueeze(1))
        cnn_features = cnn_features.permute(0, 2, 3, 1).mean(dim=2)
        cnn_features = self.syn_conv_proj(cnn_features)

        seq_lengths = src_mask.sum(dim=-1).cpu()
        lstm_features, _ = self.syn_lstm(mlm_hidden, seq_lengths)
        lstm_features = self.syn_lstm_proj(lstm_features)

        attn_cnn = self.syn_attention(aspect_hidden, cnn_features, relation_bias, src_mask)
        attn_lstm = self.syn_attention(aspect_hidden, lstm_features, relation_bias, src_mask)
        syn_context = torch.cat((attn_cnn, attn_lstm), dim=-1)
        syn_context = syn_context.view(syn_context.size(0), syn_context.size(1), 2, self.opt.bert_dim).mean(dim=2)
        return masked_average(syn_context, src_mask)

    def _build_sem_representation(self, roberta_hidden, sem_adj, aspect_mask):
        sem_hidden = F.relu(self.sem_gcn(roberta_hidden, sem_adj))
        sem_hidden = self.dropout(sem_hidden)
        return masked_average(sem_hidden, aspect_mask)

    def forward(self, inputs):
        text_bert_indices, text_prompt_indices, aspect_bert_indices, adj_matrix, edge_adj, distance_adj, relation_adj, src_mask, aspect_mask = inputs

        mlm_logits = self.mlm_model(text_bert_indices).logits
        mlm_hidden = self.dropout(self.mlm_hidden(mlm_logits))

        roberta_hidden = self.roberta_model(text_bert_indices).last_hidden_state
        roberta_hidden = self.dropout(self.layer_norm(roberta_hidden))
        aspect_hidden = self.roberta_model(aspect_bert_indices).last_hidden_state

        syn_bias = distance_adj + relation_adj.sum(dim=1)
        syn_bias = syn_bias.masked_fill(torch.isinf(adj_matrix), 0.0)
        syn_adj = normalize_adjacency(syn_bias, src_mask)
        relation_bias = syn_adj + relation_adj[:, 4]

        sem_adj = normalize_adjacency(edge_adj.float(), src_mask)

        syn_repr = self._build_syn_representation(mlm_hidden, aspect_hidden, syn_adj, relation_bias, src_mask)
        sem_repr = self._build_sem_representation(roberta_hidden, sem_adj, aspect_mask)

        p_syn = F.softmax(self.kl_syn_proj(syn_repr), dim=-1)
        p_sem = F.softmax(self.kl_sem_proj(sem_repr), dim=-1)
        kl_loss = F.kl_div(p_syn.log(), p_sem, reduction='batchmean') * self.opt.gama

        return syn_repr, sem_repr, kl_loss


class GCNBertClassifier(nn.Module):
    def __init__(self, bert, opt):
        super().__init__()
        self.opt = opt
        self.encoder = DAGFEncoder(bert, opt)

        self.syn_aux = nn.Linear(opt.bert_dim, opt.polarities_dim)
        self.sem_aux = nn.Linear(opt.bert_dim, opt.polarities_dim)
        self.syn_proj = nn.Linear(opt.bert_dim, opt.bert_dim)
        self.sem_proj = nn.Linear(opt.bert_dim, opt.bert_dim)
        self.gate = nn.Sequential(
            nn.Linear(opt.bert_dim * 2 + 2, opt.bert_dim),
            nn.ReLU(),
            nn.Linear(opt.bert_dim, opt.bert_dim),
            nn.Sigmoid(),
        )
        self.classifier = nn.Linear(opt.bert_dim, opt.polarities_dim)
        self.final_dropout = nn.Dropout(opt.final_dropout)

    @staticmethod
    def _batch_minmax(error_values):
        min_val = error_values.min(dim=0, keepdim=True).values
        max_val = error_values.max(dim=0, keepdim=True).values
        return (error_values - min_val) / (max_val - min_val + 1e-8)

    def _agf_forward(self, syn_repr, sem_repr, labels):
        y_onehot = F.one_hot(labels, num_classes=self.opt.polarities_dim).float()

        syn_pred = F.softmax(self.syn_aux(syn_repr), dim=-1)
        sem_pred = F.softmax(self.sem_aux(sem_repr), dim=-1)
        esyn = torch.norm(y_onehot - syn_pred, p=2, dim=1, keepdim=True)
        esem = torch.norm(y_onehot - sem_pred, p=2, dim=1, keepdim=True)
        esyn_norm = self._batch_minmax(esyn)
        esem_norm = self._batch_minmax(esem)

        gate_input = torch.cat((esyn_norm, esem_norm, syn_repr, sem_repr), dim=-1)
        gate = self.gate(gate_input)

        if torch.is_grad_enabled():
            grad_sem = torch.autograd.grad(esem_norm.sum(), sem_repr, retain_graph=True, create_graph=True)[0]
            h_au = grad_sem * grad_sem
        else:
            h_au = torch.zeros_like(sem_repr)

        syn_prime = F.softmax(self.syn_proj(syn_repr), dim=-1)
        sem_prime = F.softmax(self.sem_proj(sem_repr), dim=-1)
        fused_hidden = self.opt.alpha * syn_prime + self.opt.beta * sem_prime + gate * h_au
        return self.classifier(self.final_dropout(fused_hidden))

    def forward(self, inputs, labels=None):
        syn_repr, sem_repr, kl_loss = self.encoder(inputs)
        if labels is None:
            fused_hidden = self.opt.alpha * syn_repr + self.opt.beta * sem_repr
            logits = self.classifier(self.final_dropout(fused_hidden))
            return logits, kl_loss

        if torch.is_grad_enabled():
            logits = self._agf_forward(syn_repr, sem_repr, labels)
            return logits, kl_loss

        with torch.enable_grad():
            syn_repr = syn_repr.detach().requires_grad_(True)
            sem_repr = sem_repr.detach().requires_grad_(True)
            logits = self._agf_forward(syn_repr, sem_repr, labels)
        return logits.detach(), kl_loss.detach()
