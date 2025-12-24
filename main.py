import torch
import argparse
from torch.utils.tensorboard import SummaryWriter
import LTCMambaNet
from batch_gen import BatchGenerator
from trainer import Trainer
import random
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=0, type=int, help='which gpu to use, default 0')
parser.add_argument('--action', default='train', type=str)
parser.add_argument('--epo', default=200, type=int, help='epochs to train and val, default 25')
parser.add_argument('--lr', default=0.0005, type=float, help='learning rate for optimizer, default 5e-5')
parser.add_argument('--weightdecay', default=1e-5, type=float, help='weight decay for sgd, default 0')
parser.add_argument('--work', default=2, type=int, help='num of workers to use, default 4')
parser.add_argument('--num_classes', default=8, type=int, help='')
parser.add_argument('--model_path', default='', type=str, help='')
parser.add_argument('--results_dir', default='', type=str, help='')
parser.add_argument('--mamba_type', default='mamba2', type=str, help='')
parser.add_argument('--NUM_LAYERS', default=10, type=int, help='')
parser.add_argument('--NUM_STAGES', default=4, type=int, help='')
parser.add_argument('--INPUT_DIM', default=2048, type=int, help='')
parser.add_argument('--WINDOWED_Mamba_W', default=64, type=int, help='')
parser.add_argument('--LONG_TERM_Mamba_G', default=64, type=int, help='')
parser.add_argument('--numhead', default=1, type=int, help='')
args = parser.parse_args()


epochs = args.epo
num_workers = args.work
learning_rate = args.lr
weight_decay = args.weightdecay
num_classes = args.num_classes
num_gpu = torch.cuda.device_count()
use_gpu = torch.cuda.is_available()
device = torch.device("cuda:{}".format(args.gpu) if use_gpu else "cpu")

seed = random.randint(0, 2 ** 32 - 1)
# seed = 12345698
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

features_path = './dataset/ESD385/ResNet50_features/'
gt_path = './dataset/ESD385/groundTruth/'
train_vid_list_file = './dataset/ESD385/splits/train.csv'
val_vid_list_file = './dataset/ESD385/splits/val.csv'
test_vid_list_file = './dataset/ESD385/splits/test.csv'
mapping_file = "./dataset/ESD385/mapping.txt"

file_ptr = open(mapping_file, 'r')
actions = file_ptr.read().split('\n')[:-1]
file_ptr.close()
actions_dict = dict()
for a in actions:
    actions_dict[a.split()[1]] = int(a.split()[0])

model = LTCMambaNet.Temporal_mamba(args)
if args.action == 'train':
    path = ('./' + str(args.action) + '/log/' + 'LAYERS_' +
            str(args.NUM_LAYERS) + '_STAGES_' + str(args.NUM_STAGES) + '_WINDOWED_Mamba_W_' +
            str(args.WINDOWED_Mamba_W) + '_LONG_TERM_Mamba_G_' + str(args.LONG_TERM_Mamba_G) +
            '_mamba_type_' + str(args.mamba_type) + '_numhead_' + str(args.numhead) +
            '_lr_' + str(args.lr) + '_weightdecay_' + str(args.weightdecay) + '_seed_' + str(seed) + '/')
    writer = SummaryWriter(path)
    save_model_path = ('./' + str(args.action) + '/pth/' + 'LAYERS_' +
                        str(args.NUM_LAYERS) + '_STAGES_' + str(args.NUM_STAGES) + '_WINDOWED_Mamba_W_' +
                        str(args.WINDOWED_Mamba_W) + '_LONG_TERM_Mamba_G_' + str(args.LONG_TERM_Mamba_G) +
                        '_mamba_type_' + str(args.mamba_type) + '_numhead_' + str(args.numhead) +
                        '_lr_' + str(args.lr) + '_weightdecay_' + str(args.weightdecay) + '_seed_' + str(seed) + '/')
else:
    args.results_dir = ('./result' + '/LAYERS_' +
                        str(args.NUM_LAYERS) + '_STAGES_' + str(args.NUM_STAGES) + '_WINDOWED_Mamba_W_' +
                        str(args.WINDOWED_Mamba_W) + '_LONG_TERM_Mamba_G_' + str(args.LONG_TERM_Mamba_G) +
                        '_mamba_type_' + str(args.mamba_type) + '_numhead_' + str(args.numhead) +
                        '_lr_' + str(args.lr) + '_weightdecay_' + str(args.weightdecay) + '/')
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

trainer = Trainer()
if args.action == 'train':
    train_batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path)
    train_batch_gen.read_data(train_vid_list_file)
    val_batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path)
    val_batch_gen.read_data(val_vid_list_file)
    test_batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path)
    test_batch_gen.read_data(test_vid_list_file)
    trainer.train(writer, model, save_model_path, train_batch_gen, val_batch_gen, test_batch_gen, epochs, 1, learning_rate, device, weight_decay, num_classes)
elif args.action == 'test':
    trainer.predict(model, args.model_path, args.results_dir, features_path, test_vid_list_file, actions_dict, device)