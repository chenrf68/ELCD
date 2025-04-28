import numpy as np
import torch
import os
import pickle
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
from synthetic import simulate_lorenz_96, simulate_var
from utils import build_flags, time_split, save_result, evaluate_result, count_accuracy, est_causality, get_decoder_adj
from trainNet_change import train_decoder_net, train_encoder_net

# root_folder = '/home/jing_xuzijian/crf/Intrer_VAE_result/Lorenz96.20.500/seed'
def experiment(args, seed, root_folder, data, lag=2):
    #参数初始化部分
    args.seed = seed
    args.root_folder = root_folder + str(args.seed)
    if not os.path.exists(args.root_folder):
        os.makedirs(args.root_folder)
    meta_file = os.path.join(args.root_folder, 'metadata.pkl')
    pickle.dump({'args': args}, open(meta_file, "wb"))
#args.root_folder = '/home/jing_xuzijian/crf/Intrer_VAE_result/Lorenz96.20.500/seed0'

    #生成数据
    if data == 'lorenz96':
        X_np, GC = simulate_lorenz_96(p=args.num_nodes, F=10, T=args.time_length, seed=args.seed)
        #X_np = (X_np - X_np.min()) / (X_np.max() - X_np.min())

    elif data == 'kuramoto':
        from kuramoto import simulate_kuramoto
        X_np, GC = simulate_kuramoto(num_atoms=args.num_nodes, num_timesteps=args.time_length, undirected=True)
    else:
        X_np, beta, GC = simulate_var(p=args.num_nodes, T=args.time_length, lag=lag, sparsity=0.2)


    X_np_ori = X_np
    X_np = X_np.transpose(1, 0)
    X_np = X_np[:, :, np.newaxis]
    X_np = np.array(time_split(X_np, step=10))
    X_np = torch.FloatTensor(X_np)
    data = X_np
    data_loader = DataLoader(data, batch_size=args.batch_size)
    
    # 训练模型
    train_decoder_net(data_loader=data_loader, n_in=args.dims, n_hid=args.decoder_hidden, time_split=args.time_step - 1,
                      num_node=args.num_nodes, num_epoch=args.decoder_epochs, do_prob=args.decoder_dropout, lr=args.decoder_lr,
                      weight_decay=args.weight_decay, alpha=args.decoder_alpha, save_folder=args.root_folder)
    train_encoder_net(data_loader=data_loader, n_in_encoder=args.dims, n_hid_encoder=args.encoder_hidden,
                      time_split=args.time_step - 1, num_node=args.num_nodes, do_prob_encoder=args.encoder_dropout,
                      alpha_encoder=args.encoder_alpha, num_epoch=args.encoder_epochs, lr=args.encoder_lr,
                      weight_decay=args.weight_decay, sparsity_type=args.sparsity_type, divergence_type=args.divergence_type,
                      beta_sparsity=args.beta_sparsity, beta_kl=args.beta_kl, beta_mmd=args.beta_mmd, save_folder=args.root_folder,
                      n_in_decoder=args.dims, n_hid_decoder=args.decoder_hidden, do_porb_decoder=args.decoder_dropout,
                      alpha_decoder=args.decoder_alpha)
    
    adj_encoder = est_causality(X_np[:, :, :-1, :], args.dims, args.encoder_hidden, args.time_step - 1, args.num_nodes,
                            args.encoder_dropout, args.encoder_alpha, args.root_folder)
    adj_encoder = adj_encoder.mean(dim=3).mean(dim=0)
    result_encoder, _ = evaluate_result(GC, adj_encoder.detach().numpy(), args.threshold)
    save_result(result_encoder, 'encoder', args.root_folder)
    print(result_encoder)

    adj_decoder = get_decoder_adj(args.root_folder, args.dims, args.decoder_hidden, args.time_step - 1,
                                  args.num_nodes, args.decoder_dropout, args.decoder_alpha)
    adj_decoder = torch.cat([temp.unsqueeze(0) for temp in adj_decoder], dim=0)
    result_decoder, _ = evaluate_result(GC, adj_decoder.detach().numpy(), args.threshold)
    save_result(result_decoder, 'decoder', args.root_folder)
    print(result_decoder)
    
     # 对比实验
    from causallearn.search.FCMBased import lingam
    VARLinGAM = lingam.VARLiNGAM()
    VARLinGAM.fit(X_np_ori)
    adj_varlingam = VARLinGAM.adjacency_matrices_[-1]
    # Lorenz96实验参数
    adj_varlingam[adj_varlingam < 0.05] = 0
    # VAR实验参数
    # adj_varlingam[adj_varlingam < 0.1] = 0
    result_varlingam, _ = evaluate_result(GC, adj_varlingam, args.threshold)
    save_result(result_varlingam, 'varlingam', args.root_folder)
    print(result_varlingam)

    import statsmodels.api as sm
    var_ori = sm.tsa.VARMAX(X_np_ori)
    var = var_ori.fit(maxiter=1000, disp=False)
    adj_var = var.coefficient_matrices_var[0]
    # Lorenz96实验参数
    adj_var[adj_var < 0.05] = 0
    # VAR实验参数
    # adj_var[adj_val < 0.1] = 0
    result_var, _ = evaluate_result(GC, adj_var, args.threshold)
    save_result(result_var, 'var', args.root_folder)
    print(result_var)

    from contrast_models.cmlp import cMLP, train_model_ista
    X = torch.FloatTensor(X_np_ori[np.newaxis])
    X = X.cuda()
    cmlp = cMLP(X.shape[-1], lag=1, hidden=[100])
    cmlp = cmlp.cuda()

    
    # Lorenz96实验设置
    train_loss_list = train_model_ista(
        cmlp, X, lam=10, lam_ridge=1, lr=1e-3, penalty='H', max_iter=4000,
        check_every=1000)

    '''
    # VAR实验设置
    train_loss_list = train_model_ista(
        cmlp, X, lam=1e-3, lam_ridge=1e-2, lr=1e-1, penalty='H', max_iter=10000,
        check_every=1000)
    '''


    adj_cmlp = cmlp.GC(threshold=False).cpu().data.numpy()
    result_cmlp, _ = evaluate_result(GC, adj_cmlp, 0)
    save_result(result_cmlp, 'cmlp', args.root_folder)
    print(result_cmlp)

    from contrast_models.clstm import cLSTM, train_model_ista
    clstm = cLSTM(X.shape[-1], hidden=100).cuda()

    
    # Lorenz96实验设置
    train_loss_list = train_model_ista(
        clstm, X, context=10, lam=10, lam_ridge=1e-2, lr=1e-3, max_iter=4000,
        check_every=1000)

    '''
    # VAR实验设置
    train_loss_list = train_model_ista(
        clstm, X, context=10, lam=2e-2, lam_ridge=1e-1, lr=1e-3, max_iter=10000,
        check_every=1000)
    '''


    adj_clstm = clstm.GC(threshold=False).cpu().data.numpy()
    result_clstm, _ = evaluate_result(GC, adj_clstm, 0)
    save_result(result_clstm, 'clstm', args.root_folder)
    print(result_clstm)

    return result_encoder, result_decoder, result_varlingam, result_var, result_cmlp, result_clstm