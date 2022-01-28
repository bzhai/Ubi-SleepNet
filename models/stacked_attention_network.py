import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np


class SANTimeDimMatrixAttOnModality1NLayer1Concat(nn.Module):
    def __init__(self, input_feature_dim=512, attention_dim=256, time_step_dim=12):
        """
        The original paper set the attention latent space dim to be 0.5 of the feature map dimension
        """
        super(SANTimeDimMatrixAttOnModality1NLayer1Concat, self).__init__()

        self.input_feature_dim = input_feature_dim,
        self.latent_dim = attention_dim
        self.modality_1 = nn.Linear(input_feature_dim, attention_dim)  # 512 - > 256
        self.modality_2 = nn.Linear(input_feature_dim, attention_dim)  # 512 -> 256
        self.attention = nn.Linear(attention_dim, time_step_dim)  # 256 -> 12

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        self.__initial_weight()

    def __initial_weight(self):
        nn.init.xavier_uniform_(self.modality_2.weight)
        nn.init.xavier_uniform_(self.modality_1.weight)
        nn.init.xavier_uniform_(self.attention.weight)

    def forward(self, v_modality_1, v_modality_2):
        """
        :param v_modality_2: e.g. cardiac CNN final layer output matrix [batch, 512, 15]
        :param v_modality_1: e.g. activity count CNN final layer output matrix [batch, 512, 15]
        """
        # attention layer 1, car_rep: cardiac representation $$W_{cardiac,A}v_cardiac$$
        #
        mod_1_rep_l_1 = self.modality_1(torch.transpose(v_modality_1, 1, 2))
        mod_2_rep_l_1 = self.modality_2(torch.transpose(v_modality_2, 1, 2))
        h_A_l_1 = self.tanh(mod_2_rep_l_1 + mod_1_rep_l_1)  # $$h_A$$ at layer one
        p_att = self.attention(h_A_l_1)
        p_I = torch.transpose(self.softmax(p_att), 1, 2)  # here p_I is size of [t_step x t_step] after T column sum = 1
        mod_1_rep_hat = torch.matmul(v_modality_1, p_I)
        u = torch.cat([mod_1_rep_hat, v_modality_2], dim=1)  # refer to original paper's u = v^hat + v_Q if mod_1 is v
        return u

