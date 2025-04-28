from utils import build_flags, summary_result
from newexperiment import experiment
import pandas as pd
import numpy as np
import os

# Lorenz96的参数
parser = build_flags()
args = parser.parse_args(args=[])
args.seed = 2
args.num_nodes = 10
args.dims = 1
args.threshold = 0.5
args.time_length = 250
args.time_step = 10
args.decoder_epochs = 3000
args.encoder_epochs = 3000
args.batch_size = 128
args.decoder_lr = 1e-3
args.encoder_lr = 1e-3
args.weight_decay = 1e-3
args.encoder_alpha = 0.02
args.decoder_alpha = 0.04
args.beta_sparsity = 0.25 #0.25   #log_sum
args.beta_kl = 0.1        #JS散度
args.beta_mmd = 0.5      #MMD
args.encoder_hidden = 20
args.decoder_hidden = 15
args.encoder_dropout = 0.1
args.decoder_dropout = 0.2

nodes = [10]
length = [250]

for num_nodes in nodes:
    for time_length in length:
                
        args.num_nodes = num_nodes
        args.time_length = time_length

        root_fodler = r'/home/jing_xuzijian/crf/Intrer_VAE_result/Lorenz96.' + str(args.num_nodes) + '.' + str(args.time_length) + '/mmd' + str(args.beta_mmd) + '/sparsity' + str(args.beta_sparsity) + '/seed'
        root = '/home/jing_xuzijian/crf/Intrer_VAE_result/Lorenz96.' + str(args.num_nodes) + '.' + str(args.time_length)

        seeds = [2]
        encoder = []
        decoder = []
        varlingam = []
        var = []
        cmlp = []
        clstm = []

        for seed in seeds:
            result_encoder, result_decoder, result_varlingam, result_var, result_cmlp, result_clstm = experiment(args, seed,
                                                                                                                root_fodler,
                                                                                                                'lorenz96')
            encoder.append(result_encoder)
            decoder.append(result_decoder)
            varlingam.append(result_varlingam)
            var.append(result_var)
            cmlp.append(result_cmlp)
            clstm.append(result_clstm)

        results = [encoder, decoder, varlingam, var, cmlp, clstm]
        means = []
        stds = []
        for result in results:
            mean, std = summary_result(result)
            means.append(mean)
            stds.append(std)


        means = pd.DataFrame(np.array(means), columns=['accuracy', 'precision', 'recall', 'F1', 'ROC_AUC', 'PR_AUC'],
                            index=['gac', 'val', 'varlingam', 'var', 'cmlp', 'clstm'])
        stds = pd.DataFrame(np.array(stds), columns=['accuracy', 'precision', 'recall', 'F1', 'ROC_AUC', 'PR_AUC'],
                            index=['gac', 'val', 'varlingam', 'var', 'cmlp', 'clstm'])
        means.to_excel(os.path.join(root, 'mean.xlsx'))
        stds.to_excel(os.path.join(root, 'std.xlsx'))