# main file for the project
import os
import argparse
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
parser.add_argument('--net_sub_suffix',type=str,default="SegJointGene_CID",help='')
parser.add_argument('--if_load_ckpt',type=str2bool,default=False,help='set true for loading net from trained before')
parser.add_argument('--ckpt_load_epoch',type=int,default=30,help='the epoch of loaded net')
# add computing args
parser.add_argument('--gpu_id',type=int,default=0,help='')
parser.add_argument('--num_workers',type=int,default=14,help='')
parser.add_argument('--random_seed',type=int,default=1234,help='')
parser.add_argument('--save_space_trick',type=str2bool,default=False,help='if use trick for save space')
parser.add_argument('--save_space_trick_epoch_num',type=int,default=1,help='interval of saving space')
parser.add_argument('--save_csv_epoch_interval',type=int,default=5,help='interval of saving logger in csv')
# add preprocess arg
parser.add_argument('--global_scale',type=int,default=1,help='Max size: 2*1000=2000. 4000 may get memory error')
parser.add_argument('--patch_size',type=int,default=256,help='size for each patch')
parser.add_argument('--density_sigma',type=float,default=5,help='for density map generation')
# add CA1 arg
parser.add_argument('--CA1_sub_path',type=str,default='3_1_left')
# add SegJointGene arg
parser.add_argument('--attr_method',type=str,default='CID',help='CID, IG, none')
parser.add_argument('--attr_n_gene',type=int,default=50,help='')
parser.add_argument('--attr_n_celltype',type=int,default=59,help='')
parser.add_argument('--attr_epoch',type=int,default=30,help='')
parser.add_argument('--pixel_distance',type=int,default=5,help='')
parser.add_argument('--expand_k',type=int,default=3,help='')
parser.add_argument('--prediction_threshold',type=float,default=0.95,help='')
# add CID arg
parser.add_argument('--CID_n_steps',type=int,default=20,help='')
parser.add_argument('--CID_lr',type=float,default=0.1,help='')
parser.add_argument('--CID_lambda_param',type=float,default=0.01,help='')
parser.add_argument('--CID_beta',type=int,default=1.2,help='')
parser.add_argument('--CID_gene_chunk_size',type=int,default=50,help='')
parser.add_argument('--CID_cell_chunk_size',type=int,default=10,help='')

# get args from parser
args = parser.parse_args()
# add args based on dataset
if args.datasets_name == 'CA1':
    args.input_channel = 84
    args.output_channel = 59
    args.patch_size = 256
    args.net_batch_size = 8
# add args based on net
args.net_epoch = 200
args.net_optimizer = 'Adam'
args.net_weight_decay = 0
args.net_lr = 1e-4
# add args based on SegJointGene
if args.datasets_name == 'CA1':
    args.attr_n_celltype = args.output_channel
    args.attr_n_gene = 50
    args.CID_gene_chunk_size = args.attr_n_gene
    args.CID_cell_chunk_size = 10
# running experiments by step name!
if args.step_name == 'preprocess_CA1':
    from step_preprocess_CA1 import step_preprocess_CA1
    step_preprocess_CA1(root_path, args)
elif args.step_name == 'SegJointGene':
    from step_SegJointGene import step_SegJointGene
    step_SegJointGene(root_path, args)
else:
    raise NameError('Can not recognize the name of step')
