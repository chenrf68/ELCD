import torch 
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data, gain=1.414)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()  

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_features]
        x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        return self.batch_norm(x)
    

class GATLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GATLayer, self).__init__()
        self.in_features = in_features  # 节点向量的特征维度
        self.out_features = out_features  # 经过GAT之后的特征维度
        self.dropout = dropout  # dropout参数
        self.alpha = alpha  # leakyReLU的参数

        # 定义可训练参数，即论文中的W和a
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # xavier初始化
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)  # xavier初始化

        # 定义leakyReLU激活函数
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        # 定义softplus函数
        # self.soft = nn.Softplus()


    def forward(self, input_h, adj):
        """
        input_h:  [B, N, in_features]
        adj: 图的邻接矩阵 维度[N, N] 非零即一，可以参考5分钟-通俗易懂-图神经网络计算
        """
        # self.W [in_features,out_features]
        # input_h × self.W  ->  [B, N, out_features]

        h = torch.bmm(input_h,  torch.repeat_interleave(self.W.unsqueeze(0), input_h.size(0), dim=0))  # [B, N, out_features]

        N = h.size()[1]  # N 图的节点数
        input_concat = torch.cat([h.repeat(1, 1, N).view(input_h.size(0), N * N, -1), h.repeat(1, N, 1)], dim=2). \
            view(input_h.size(0), N, -1, 2 * self.out_features) 

        # [B, N, N, 2*out_features]
        e = self.leakyrelu(torch.matmul(input_concat, self.a).squeeze(3))
        # e = self.soft(torch.matmul(input_concat, self.a).squeeze(3))
        # [B, N, N, 1] => [B, N, N] 图注意力的相关系数（未归一化）

        # zero_vec = -1e12 * torch.ones_like(e)  # 将没有连接的边置为负无穷
        # zero_vec = torch.zeros_like(e)  # 将没有连接的边置为0
        # attention = torch.where(adj > 0, e, zero_vec)  # [B, N, N]
        # 表示如果邻接矩阵元素大于0时，则两个节点有连接，该位置的注意力系数保留，
        # 否则需要mask并置为非常小的值，原因是softmax的时候这个最小值会不考虑。
        attention = torch.repeat_interleave(adj.unsqueeze(0), input_h.size(0), dim=0) #* e
        #attention = F.softmax(attention, dim=2)  # softmax形状保持不变 [B, N, N]，得到归一化的注意力权重！
        # attention = F.dropout(attention, self.dropout, training=self.training)  # dropout，防止过拟合
        output_h = torch.bmm(attention, h)  # [B, N, N].[B, N, out_features] => [B, N, out_features]
        # 得到由周围节点通过注意力权重进行更新的表示
        return output_h

'''
class encoder(nn.Module):
    def __init__(self, n_in, n_hid, n_out, n_node, do_prob, alpha):
        super(encoder, self).__init__()
        self.lstm = nn.LSTM(n_in, n_hid, batch_first=True)
        self.mlp = MLP(n_hid, 2 * n_hid, n_hid, do_prob)
        self.gat = GATLayer(n_hid, n_hid, do_prob, alpha)
        self.adj = nn.Parameter(torch.ones([n_node, n_node]))
        self.bn = nn.BatchNorm1d(n_hid)
        self.fc_mu = nn.Linear(n_hid, n_out)
        self.fc_std = nn.Linear(n_hid, n_out)
        # self.gamma_fc_mu = nn.Linear(n_hid, n_out)
        # self.gamma_fc_std = nn.Linear(n_hid, n_out)
        # self.theta_fc_mu = nn.Linear(n_hid, n_out)
        # self.theta_fc_std = nn.Linear(n_hid, n_out)


    def batch_norm(self, inputs):
        x = inputs.reshape(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.reshape(inputs.size(0), inputs.size(1), -1)

    # 静态图时候的情况
    def forward(self, inputs):
        # inputs: [num_sample 128, num_node 20, num_timepoint 9, num_feature 1]
        x = inputs.reshape(inputs.size(0) * inputs.size(1), inputs.size(2), inputs.size(3))
        # x: [num_sample * num_node, num_timepoint, num_feature]
        x, _ = self.lstm(x)
        # x: [num_sample * num_node, num_timepoint, num_hid]
        x = self.bn(x[:, -1, :])
        # x: [num_sample * num_node, num_hid]
        x = x.reshape(inputs.size(0), inputs.size(1), -1)
        # x: [num_sample, num_node, num_hid]
        x = self.gat(x, self.adj)
        # x: [num_sample 128, num_node 20, num_hid 15]
        x = self.mlp(x)
        # x: [num_sample 128, num_node 20, num_hid 15]
        mu = self.fc_mu(x)
        # mu: [num_sample 128, num_node 20, num_out 1]
        std = self.fc_std(x)
        # std: [num_sample 128, num_node 20, num_out 1]
        return mu, std
'''


#动态时候使用的代码
class encoder(nn.Module):
    def __init__(self, n_in, n_hid, n_out, n_time, n_node, do_prob, alpha):
        super(encoder, self).__init__()
        self.lstm = nn.LSTM(n_in, n_hid, batch_first=True)
        self.mlp = MLP(n_hid, 2 * n_hid, n_hid, do_prob)
        self.bn = nn.BatchNorm1d(n_hid)
        self.graph_kernel = nn.Parameter(torch.ones([n_node, n_node]))
        self.fc_mu = nn.Linear(n_hid, n_out)
        self.fc_std = nn.Linear(n_hid, n_out)

    def batch_norm(self, inputs):
        x = inputs.reshape(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.reshape(inputs.size(0), inputs.size(1), -1)

    def graph_conv(self, inputs):
        x = inputs.reshape(inputs.size(0), inputs.size(1), -1)
        A = torch.repeat_interleave(self.graph_kernel.unsqueeze(0), inputs.size(0), dim=0)
        x = torch.bmm(A, x)
        return x.reshape(inputs.size(0), -1, inputs.size(3))
    
    # 动态图时候的情况
    def forward(self, inputs):
        # inputs: [num_sample 128, num_node 20, num_timepoint 9, num_feature 1]
        x = inputs.reshape(inputs.size(0) * inputs.size(1), inputs.size(2), inputs.size(3))
        # x: [num_sample * num_node, num_timepoint, num_feature]
        x, _ = self.lstm(x)
        # x: [num_sample * num_node, num_timepoint, num_hid]
        x = self.batch_norm(x)
        # x: [num_sample * num_node, num_timepoint, num_hid]
        x = x.reshape(inputs.size(0), inputs.size(1), inputs.size(2), -1)
        # x: [num_sample, num_node, num_timepoint, num_hid]
        x = self.graph_conv(x)
        # x: [num_sample 128, num_node 20, num_timepoint *num_hid 135]
        x = x.reshape(inputs.size(0), inputs.size(1)*inputs.size(2), -1)
        # x: [num_sample 128, num_node * num_timepoint 180, num_hid 15]
        x = self.mlp(x)
        # x: [num_sample 128, num_node * num_timepoint 180, num_hid 15]
        mu = self.fc_mu(x)
        # mu: [num_sample 128, num_node*num_timepoint 180, num_out 1]
        mu = mu.reshape(inputs.size(0), inputs.size(1), inputs.size(2), -1)
        # mu: [num_sample 128, num_node 20, num_timepoint 9, num_out 1]
        std = self.fc_std(x)
        # std: [num_sample 128, num_node*num_timepoint 180, num_out 1]
        std = std.reshape(inputs.size(0), inputs.size(1), inputs.size(2), -1)
        # std: [num_sample 128, num_node 20, num_timepoint 9, num_out 1]
        return mu, std
    

class decoder(nn.Module):
    def __init__(self, n_in, n_hid, n_time, n_node, do_prob, alpha):
        super(decoder, self).__init__()
        #单步预测暂时不用lstm，如果效果不好，则可以加入试一试
        # self.lstm = nn.LSTM(n_in, n_hid, batch_first=True)
        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
        self.mlp2 = MLP(n_hid, 2 * n_hid, n_hid, do_prob)
        self.graph_kernel = nn.Parameter(torch.ones([1, n_node]))
        self.fc = nn.Linear(n_hid, n_in)
        # self.add = nn.Parameter(torch.ones([1, n_node]))
    
    def graph_conv(self, inputs):
        # 注意这里的A是目标结点在邻接矩阵中所对应的行
        x = inputs.reshape(inputs.size(0), inputs.size(1), -1)
        A = torch.repeat_interleave(self.graph_kernel.unsqueeze(0), inputs.size(0), dim=0)
        x = torch.bmm(A, x)

        return x.reshape(inputs.size(0), inputs.size(2), inputs.size(3))


    def forward(self, inputs, idx):
        # inputs: [num_sample 128, num_node 20, num_timepoint 9, num_feature 1]
        x = inputs.reshape(inputs.size(0), -1, inputs.size(3)).to(inputs.device)
        # x: [num_sample 128, num_node * num_timepoint 180, num_feature 1]
        # x, _ = self.lstm(x)
        # x: [num_sample 128, num_node * num_timepoint 180, num_hid 15]
        x = self.mlp1(x)
        # x: [num_sample 128, num_node * num_timepoint 180, num_hid 15]
        x = x.reshape(inputs.size(0), inputs.size(1), inputs.size(2), -1)
        # x: [num_sample 128, num_node 20, num_timepoint 9, num_hid 15]
        x = self.graph_conv(x)
        # x: [num_sample 128, num_node 20, num_hid 135]
        x = self.mlp2(x)
        # x: [num_sample 128, num_timepoint 9, num_hid 15]
        x = self.fc(x)
        # x: [num_sample 128, num_timepoint 9, num_feature 1]
        # input[128,20,9,1]->h1[128,180,1]->h2[128,180,15]->h3[128,20,135](inputs of GAT)
        # ->h4[128,i,135](outputs of GAT)->h5[128,1,135](output of bmm)->h6[128,9,15]->
        # h7[128,9,15]->output[128,9,1]
        return x.reshape(inputs.size(0), inputs.size(2), inputs.size(3))

'''    
#对于静态的情况而言    
def mask_inputs(mask, inputs):
    # mask: [num_sample 128, num_node 20, num_feature 1]
    # inputs: [num_sample 128, num_node 20, num_timepoint 9, num_feature 1]
    # return: [num_sample 128, num_node 20, num_timepoint 9, num_feature 1]
    mask = torch.repeat_interleave(mask.unsqueeze(2), inputs.size(2), dim=2)
    return inputs * mask

'''
#动态的代码情况
def mask_inputs(mask, inputs):
    # mask: [num_sample 128, num_node 20, num_timepoint 9, num_feature 1]
    # inputs: [num_sample 128, num_node 20, num_timepoint 9, num_feature 1]
    # return: [num_sample 128, num_node 20, num_timepoint 9, num_feature 1]
    inputs = torch.repeat_interleave(mask, inputs.size(-1), 3) * inputs

    return inputs


