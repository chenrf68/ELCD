from Model import *
from utils import loss_sparsity, loss_divergence
import os 
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import time
from itertools import chain


def train_idx_Inter_net(data_loader, idx, n_in, n_hid, time_split, num_node, adj, M, gamma_matrix, num_epoch, do_prob, lr, weight_decay, alpha, 
                       beta_sparsity, beta_kl, log, save_file):
    # 对单个节点训练干预神经网络
    Inter_encoder = encoder(n_in, n_hid, n_in, time_split, num_node, do_prob, alpha)
    Inter_encoder = Inter_encoder.cuda()
    Inter_decoder = decoder(n_in, n_hid, time_split, num_node, do_prob, alpha)
    Inter_decoder = Inter_decoder.cuda()
    optimizer = optim.Adam(params = chain(Inter_encoder.parameters(), Inter_decoder.parameters()), lr=lr, weight_decay=weight_decay)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
    loss_mse = nn.MSELoss()
    best_loss = np.inf
    for epoch in range(num_epoch):
        t = time.time()
        Loss = []
        MSE_loss = []
        SPA_loss = []
        KL_loss = []
        #MMD_loss = []
        for batch_idx, data in enumerate(data_loader):
            optimizer.zero_grad()
            data = data.cuda()
            target = data[:, idx, 1:, :]
            inputs = data[:, :, :-1, :]

            mu, log_var = Inter_encoder(inputs)  #Inter_encoder(inputs, adj)
            sigma = torch.exp(log_var / 2)
            gamma = torch.randn(size = mu.size()).cuda()
            # theta = torch.randn(size = mu.size()).cuda()
            gamma = mu + sigma * gamma
            # theta = mu + sigma * theta
            # mask = torch.sigmoid(gamma * theta)
            mask = torch.sigmoid(gamma) #* torch.sigmoid(theta + gamma)

            
            # 静态时候的代码情况，需要进行维度降低
            # gamma = torch.softmax(gamma.squeeze().mean(dim = 0), dim=0)
            gamma = torch.sigmoid(gamma).squeeze().mean(dim = 0).mean(dim = 1)
            # theta = torch.sigmoid(theta.squeeze().mean(dim = 0))
            


        
            inputs = mask_inputs(mask, inputs)
            pred = Inter_decoder(inputs, idx)   #Inter_decoder(inputs, adj, idx)
            adj[idx, :] = torch.bernoulli(gamma)#torch.softmax(gamma + theta, dim=0)  #gamma * theta  #torch.bernoulli(gamma) * torch.bernoulli(theta) 0.5*(gamma + theta) 



            mse_loss = loss_mse(pred, target)
            
            # 最初始的直接使用VAE的的一些正则项做的情况，然后直接采用论文的了l1正则，并不能起到很好的特征选择的效果
            '''
            #静态情况下的正则项
            spa_loss = gamma.sum()#mask.sum() #+ Inter_decoder.adj[idx, :].sum() #gam
            # spa_loss = gamma.mean()
            kl_loss = ((-0.5*(1+log_var.squeeze() - mu.squeeze()**2 - torch.exp(log_var.squeeze()))).sum(dim = 1)).mean(dim = 0)
            # mmd_loss = mmd_loss_func(gamma, theta)
            '''

            '''
            # 动态情况下的正则项
            spa_loss = gamma.sum()
            kl_loss = ((-0.5*(1+log_var.squeeze() - mu.squeeze()**2 - torch.exp(log_var.squeeze()))).sum(dim = 1)).mean(dim = 0).mean(dim = 0)
            '''


            
            # 新的正则方法采用了JS散度、log-sum的情况
            spa_loss = loss_sparsity(mask, 'log_sum')
            kl_loss = loss_divergence(mask, 'JS')

            
            loss = mse_loss + beta_sparsity * spa_loss + beta_kl * kl_loss

            loss.backward()
            optimizer.step()

            Loss.append(loss.item())
            MSE_loss.append(mse_loss.item())
            SPA_loss.append(spa_loss.item())
            KL_loss.append(kl_loss.item())
            #MMD_loss.append(mmd_loss.item())
    
        if epoch == 500:
            optimizer.param_groups[0]['lr'] = lr/10

        if epoch % 100 == 0:
            print(  'Feature: {:04d}'.format(idx + 1),
                    'Epoch: {:04d}'.format(epoch),
                    'Loss: {:.10f}'.format(np.mean(Loss)),
                    'MSE_Loss: {:.10f}'.format(np.mean(MSE_loss)),
                    'Sparsity_loss: {:.10f}'.format(np.mean(SPA_loss)),
                    'KL_loss: {:.10f}'.format(np.mean(KL_loss)),
                    #'MMD_loss: {:.10f}'.format(np.mean(MMD_loss)),
                    'time: {:.4f}s'.format(time.time() - t))
            
        if np.mean(Loss) < best_loss:
            best_loss = np.mean(Loss)
            M[idx, :] = adj[idx, :]
            gamma_matrix[idx, :] = gamma
            # theta_matrix[idx, :] = theta
            torch.save({
                        'encoder_state_dict': Inter_encoder.state_dict(),
                        'decoder_state_dict': Inter_decoder.state_dict(),
                        # 'adj' : adj

                        }, save_file)
            # np.save(save_file + str(idx) + '.npy', mask.cpu().detach().numpy())

            print('Feature: {:04d}'.format(idx + 1),
                  'Epoch: {:04d}'.format(epoch),
                  'Loss: {:.10f}'.format(np.mean(Loss)),
                  'mse_loss: {:.10f}'.format(np.mean(MSE_loss)),
                  'Sparsity_loss: {:.10f}'.format(np.mean(SPA_loss)),
                  'KL_loss: {:.10f}'.format(np.mean(KL_loss)),
                  #'mmd_loss: {:.10f}'.format(np.mean(mmd_loss)),
                  'time: {:.4f}s'.format(time.time() - t), file=log)
            
        log.flush()

#save_folder = '/home/jing_xuzijian/crf/Intrer_VAE_result/Lorenz96.20.500/seed0'
def train_inter_net(data_loader, n_in, n_hid, num_node, time_split, adj, M, gamma_matrix, num_epoch, do_prob, lr, weight_decay, alpha, 
              beta_sparsity, beta_kl, save_folder):
    # 训练整个干预神经网络
    log_file = os.path.join(save_folder, 'log.txt') #log_file = '/home/jing_xuzijian/crf/Intrer_VAE_result/Lorenz96.20.500/seed0/log.txt'
    log = open(log_file, 'w')
    # 训练两次以免最开始全是0的adj有所影响
    for idx in range(num_node):
        print('Begin training feature: {:04d}'.format(idx + 1))
        save_file = 'Inter_net' + str(idx) + '.pt'
        save_file = os.path.join(save_folder, save_file)
        train_idx_Inter_net(data_loader, idx, n_in, n_hid, time_split, num_node, adj, M, gamma_matrix, num_epoch, do_prob, lr, weight_decay, alpha,
                                beta_sparsity, beta_kl, log, save_file)
        adj[idx, :] = M[idx, :]
            
    log.close()