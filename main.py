# main file for the project
import os
import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

# set root path and parser
parser = argparse.ArgumentParser('parameters')
root_path = os.getcwd()

# add baisc args
parser.add_argument('--datasets_name',type=str,default='CUB',help='')
parser.add_argument('--net_name',type=str,default='unet',help='')
parser.add_argument('--step_name',type=str,default='net',help='')
# add adjustable net args
parser.add_argument('--net_sub_suffix',type=str,default="SegjointGene",help='')
parser.add_argument('--if_load_ckpt',type=str2bool,default=False,help='set true for loading net from trained before')
parser.add_argument('--ckpt_load_epoch',type=int,default=30,help='the epoch of loaded net')
# add computing args
parser.add_argument('--gpu_id',type=int,default=0,help='')
parser.add_argument('--num_workers',type=int,default=14,help='')
parser.add_argument('--random_seed',type=int,default=1234,help='')
parser.add_argument('--save_space_trick',type=str2bool,default=False,help='if use trick for save space')
parser.add_argument('--save_space_trick_epoch_num',type=int,default=1,help='interval of saving space')
parser.add_argument('--save_csv_epoch_interval',type=int,default=5,help='interval of saving logger in csv')
# add SegjointGene arg
parser.add_argument('--patch_size',type=int,default=128,help='size for each patch')
parser.add_argument('--pixel_distance',type=int,default=5,help='')
parser.add_argument('--expand_k',type=int,default=5,help='')
parser.add_argument('--prediction_threshold',type=float,default=0.5,help='')
parser.add_argument('--attr_grid',type=str,default='2x2',help='')
parser.add_argument('--attr_epoch',type=int,default=20,help='')
# add CID arg
parser.add_argument('--CID_n_steps',type=int,default=20,help='')
parser.add_argument('--CID_lr',type=float,default=0.2,help='')
parser.add_argument('--CID_lambda_param',type=float,default=0.01,help='')
parser.add_argument('--CID_beta',type=int,default=1.2,help='')
parser.add_argument('--CID_chunk_size',type=int,default=8,help='')

# get args from parser
args = parser.parse_args()
# add args based on dataset
if args.datasets_name == 'CA1':
    args.input_channel = 84
    args.output_channel = 59
# add args based on net
args.net_epoch = 200
args.net_optimizer = 'Adam'
args.net_batch_size = 16
args.net_weight_decay = 0
args.net_lr = 1e-4

# running experiments by step name!
if args.step_name == 'SegjointGene_CID':
    from step_SegjointGene_CID import step_SegjointGene_CID
    step_SegjointGene_CID(root_path, args)
else:
    raise NameError('Can not recognize the name of step')
