from Model import *
from utils import loss_sparsity, loss_divergence, set_encoder_adj, loss_mmd
import os 
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import time


def train_idx_decoder_net(data_loader, idx, n_in, n_hid, time_split, num_node, num_epoch, do_prob,
                          lr, weight_decay, alpha, log, decoder_file):
    # 对单个节点训练decoder
    Inter_decoder = decoder(n_in, n_hid, time_split, num_node, do_prob, alpha)
    Inter_decoder = Inter_decoder.cuda()
    optimizer = optim.Adam(params = Inter_decoder.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
    loss_val = nn.MSELoss()
    best_loss = np.Inf
    for epoch in range(num_epoch):
        scheduler.step()
        t = time.time()
        Loss = []
        mse_loss = []
        for batch_idx, data in enumerate(data_loader):
            data = data.cuda()
            target = data[:, idx, 1:, :]
            optimizer.zero_grad()
            inputs = data[:, :, :-1, :]
            pred = Inter_decoder(inputs, idx)
            mse = loss_val(pred, target)
            loss = mse

            loss.backward()
            optimizer.step()


            Loss.append(loss.item())
            mse_loss.append(mse.item())

        if epoch % 100 == 0:
            print('Feature: {:04d}'.format(idx + 1),
                'Epoch: {:04d}'.format(epoch),
                'Loss: {:.10f}'.format(np.mean(Loss)),
                'MSE_Loss: {:.10f}'.format(np.mean(mse_loss)),
                'time: {:.4f}s'.format(time.time() - t))
            
        if np.mean(mse_loss) < best_loss:
            best_loss = np.mean(mse_loss)
            torch.save(Inter_decoder.state_dict(), decoder_file)
            print('Feature: {:04d}'.format(idx + 1),
                  'Epoch: {:04d}'.format(epoch),
                  'Loss: {:.10f}'.format(np.mean(Loss)),
                  'mse_loss: {:.10f}'.format(np.mean(mse_loss)),
                #   'mmd_loss: {:.10f}'.format(np.mean(mmd_loss)),
                  'time: {:.4f}s'.format(time.time() - t), file=log)
            
        log.flush()


def train_decoder_net(data_loader, n_in, n_hid, time_split, num_node, num_epoch, do_prob,
                          lr, weight_decay, alpha, save_folder):
    # 训练整个decoder网络
    log_file = os.path.join(save_folder, 'log_val.txt')
    log = open(log_file, 'w')
    for idx in range(num_node):
        print('Begin training feature: {:04d}'.format(idx + 1))
        decoder_file = 'decoder' + str(idx) + '.pt'
        decoder_file = os.path.join(save_folder, decoder_file)
        train_idx_decoder_net(data_loader, idx, n_in, n_hid, time_split, num_node, num_epoch, do_prob,
                            lr, weight_decay, alpha, log, decoder_file)
    log.close()


def train_idx_encoder_net(data_loader, idx, init_adj, n_in_encoder, n_hid_encoder, time_split, num_node, do_prob_encoder, alpha_encoder,
                          num_epoch, lr, weight_decay, sparsity_type, divergence_type, beta_sparsity, beta_kl, beta_mmd, log, 
                          encoder_file, n_in_decoder, n_hid_decoder, do_porb_decoder, alpha_decoder, decoder_file):
    
    # 对单个结点训练encoder
    Inter_decoder = decoder(n_in_decoder, n_hid_decoder, time_split, num_node, do_porb_decoder, alpha_decoder)
    Inter_decoder.load_state_dict(torch.load(decoder_file))
    Inter_decoder = Inter_decoder.cuda()
    Inter_decoder.eval()

    Inter_encoder = encoder(init_adj, n_in_encoder, n_hid_encoder, n_in_encoder, time_split, do_prob_encoder, alpha_encoder)
    Inter_encoder = Inter_encoder.cuda()

    optimizer = optim.Adam(Inter_encoder.parameters(), lr=lr, weight_decay=weight_decay)
    loss_mse = nn.MSELoss()
    best_loss = np.inf
    
    for epoch in range(num_epoch):
        t = time.time()
        Loss = []
        MSE_loss = []
        SPA_loss = []
        KL_loss = []
        MMD_loss = []
        for batch_idx, data in enumerate(data_loader):
            optimizer.zero_grad()
            data = data.cuda()
            target = data[:, idx, 1:, :]
            inputs = data[:, :, :-1, :]

            mu, log_var = Inter_encoder(inputs)  
            sigma = torch.exp(log_var / 2)
            gamma = torch.randn(size = mu.size()).cuda()
            gamma = mu + sigma * gamma
            mask = torch.sigmoid(gamma)


            inputs = mask_inputs(mask, inputs)
            pred = Inter_decoder(inputs, idx)  



            mse_loss = loss_mse(pred, target)
            spa_loss = loss_sparsity(mask, sparsity_type)
            kl_loss = loss_divergence(mask, divergence_type)
            mmd_loss = loss_mmd(data[:, :, 1:, :], pred, idx)

            loss = mse_loss + beta_sparsity * spa_loss + beta_kl * kl_loss + beta_mmd * mmd_loss

            loss.backward()
            optimizer.step()

            Loss.append(loss.item())
            MSE_loss.append(mse_loss.item())
            SPA_loss.append(spa_loss.item())
            KL_loss.append(kl_loss.item())
            MMD_loss.append(mmd_loss.item())
        
        # if epoch == 500:
        #     optimizer.param_groups[0]['lr'] = args.lr/10

        if epoch % 100 == 0:
            print(  'Feature: {:04d}'.format(idx + 1),
                    'Epoch: {:04d}'.format(epoch),
                    'Loss: {:.10f}'.format(np.mean(Loss)),
                    'MSE_Loss: {:.10f}'.format(np.mean(MSE_loss)),
                    'Sparsity_loss: {:.10f}'.format(np.mean(SPA_loss)),
                    'KL_loss: {:.10f}'.format(np.mean(KL_loss)),
                    'MMD_loss: {:.10f}'.format(np.mean(MMD_loss)),
                    'time: {:.4f}s'.format(time.time() - t))
            

        if np.mean(Loss) < best_loss:
            best_loss = np.mean(Loss)
            torch.save(Inter_encoder.state_dict(), encoder_file)

            print('Feature: {:04d}'.format(idx + 1),
                  'Epoch: {:04d}'.format(epoch),
                  'Loss: {:.10f}'.format(np.mean(Loss)),
                  'mse_loss: {:.10f}'.format(np.mean(MSE_loss)),
                  'Sparsity_loss: {:.10f}'.format(np.mean(SPA_loss)),
                  'KL_loss: {:.10f}'.format(np.mean(KL_loss)),
                  'mmd_loss: {:.10f}'.format(np.mean(MMD_loss)),
                  'time: {:.4f}s'.format(time.time() - t), file=log)
        log.flush()

def train_encoder_net(data_loader, n_in_encoder, n_hid_encoder, time_split, num_node, do_prob_encoder, alpha_encoder,
                          num_epoch, lr, weight_decay, sparsity_type, divergence_type, beta_sparsity, beta_kl, beta_mmd, save_folder,
                          n_in_decoder, n_hid_decoder, do_porb_decoder, alpha_decoder):
    
    # 训练整个encoder网络
    log_file = os.path.join(save_folder, 'log_val.txt')
    log = open(log_file, 'w')
    init_adj = set_encoder_adj(save_folder, n_in_decoder, n_hid_decoder, time_split, num_node, do_porb_decoder, alpha_decoder)

    for idx in range(num_node):
        print('Begin training feature: {:04d}'.format(idx + 1))
        encoder_file = 'encoder' + str(idx) + '.pt'
        encoder_file = os.path.join(save_folder, encoder_file)
        decoder_file = 'decoder' + str(idx) + '.pt'
        decoder_file = os.path.join(save_folder, decoder_file)
        train_idx_encoder_net(data_loader, idx, init_adj, n_in_encoder, n_hid_encoder, time_split, num_node, do_prob_encoder, alpha_encoder,
                          num_epoch, lr, weight_decay, sparsity_type, divergence_type, beta_sparsity, beta_kl, beta_mmd, log, 
                          encoder_file, n_in_decoder, n_hid_decoder, do_porb_decoder, alpha_decoder, decoder_file)
    log.close()