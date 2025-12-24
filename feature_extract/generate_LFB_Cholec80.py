import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import resnet
from dataset_Cholec80 import ImageFolderDataset
import csv
from natsort import natsorted
from timm.models import create_model

parser = argparse.ArgumentParser(description='lstm training')
parser.add_argument('--gpu', default=1, type=int, help='which gpu to use, default 0')
parser.add_argument('--val', default=200, type=int, help='valid batch size, default 10')
parser.add_argument('--work', default=4, type=int, help='num of workers to use, default 4')

args = parser.parse_args()

gpu_usg = args.gpu
val_batch_size = args.val
workers = args.work
num_gpu = torch.cuda.device_count()
use_gpu = torch.cuda.is_available()
device = torch.device("cuda:{}".format(gpu_usg) if use_gpu else "cpu")

npy_path = '/home/zhangxiangning/Cholec80/ResNet50_features/'
image_root_path = '/home/zhangxiangning/Cholec80/picture/'
label_dir = '/home/zhangxiangning/Cholec80/groundTruth/'
train_vid_list_file = '/home/zhangxiangning/Cholec80/splits_Diffusion/train.csv'
val_vid_list_file = '/home/zhangxiangning/Cholec80/splits_Diffusion/test.csv'
mapping_file = "/home/zhangxiangning/Cholec80/mapping.txt"

if not os.path.exists(npy_path):
    os.makedirs(npy_path)

file_ptr = open(mapping_file, 'r')
actions = file_ptr.read().split('\n')[:-1]
file_ptr.close()
actions_dict = dict()
for a in actions:
    actions_dict[a.split()[1]] = int(a.split()[0])

model_weight_path = './feature_extract/Cholec80/pth/temp/51.pth'
num_classes = 7
mean = [0.40774766, 0.25766712, 0.25444105]
std = [0.22692588, 0.20076735, 0.19523472]

def get_long_feature(start_index, lfb, LFB_length):
    long_feature = []
    long_feature_each = []
    # 上一个存在feature的index
    for k in range(LFB_length):
        LFB_index = (start_index + k)
        LFB_index = int(LFB_index)
        long_feature_each.append(lfb[LFB_index])
    long_feature.append(long_feature_each)
    return long_feature

def get_data(image_root_path, label_dir, train_vid_list_file, val_vid_list_file, actions_dict):

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    train_dataset = ImageFolderDataset(train_vid_list_file, image_root_path, label_dir, actions_dict, test_transforms)
    val_dataset = ImageFolderDataset(val_vid_list_file, image_root_path, label_dir, actions_dict, test_transforms)

    return train_dataset, val_dataset


g_LFB_train = np.zeros(shape=(0, 1024))
g_LFB_val = np.zeros(shape=(0, 1024))

def generate_LFB(train_dataset, val_dataset):

    global g_LFB_train
    global g_LFB_val

    print("loading features!>.........")

    train_feature_loader = DataLoader(
        train_dataset,
        batch_size=val_batch_size,
        num_workers=workers,
        shuffle=False,
        pin_memory=False
    )
    val_feature_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        num_workers=workers,
        shuffle=False,
        pin_memory=False
    )

    model_LFB = create_model('swin_base_patch4_window7_224', num_classes=num_classes)
    model_LFB.load_state_dict(torch.load(model_weight_path, map_location=device), strict=False)
    model_LFB.eval()

    if use_gpu:
        model_LFB.to(device)

    with torch.no_grad():
        for data in train_feature_loader:
            if use_gpu:
                inputs, labels_phase = data[0].to(device), data[1].to(device)
            else:
                inputs, labels_phase = data[0], data[1]

            outputs_feature = model_LFB.forward_features(inputs).data.cpu().numpy()
            g_LFB_train = np.concatenate((g_LFB_train, outputs_feature), axis=0)

            print("train feature length:", len(g_LFB_train))

        for data in val_feature_loader:
            if use_gpu:
                inputs, labels_phase = data[0].to(device), data[1].to(device)
            else:
                inputs, labels_phase = data[0], data[1]

            outputs_feature = model_LFB.forward_features(inputs).data.cpu().numpy()
            g_LFB_val = np.concatenate((g_LFB_val, outputs_feature), axis=0)

            print("val feature length:", len(g_LFB_val))

    print("finish!")
    g_LFB_train = np.array(g_LFB_train)
    g_LFB_val = np.array(g_LFB_val)

    return g_LFB_train, g_LFB_val

def generate_npy(g_LFB_train, g_LFB_val, train_list_of_examples, val_list_of_examples, train_num_each, val_num_each):

    for i in [x for x in range(len(train_list_of_examples))]:
        long_feature = get_long_feature(start_index=train_start_vidx[i],
                                        lfb=g_LFB_train, LFB_length=train_num_each[i])
        long_feature = np.array(long_feature)
        long_feature = np.squeeze(long_feature, axis=0)
        long_feature = long_feature.transpose(-1, -2)
        path_whole = train_list_of_examples[i]
        np.save(npy_path + path_whole[:-4] + '.npy', long_feature)
        print(path_whole)
        print(long_feature.shape)

    for i in [x for x in range(len(val_list_of_examples))]:
        long_feature = get_long_feature(start_index=val_start_vidx[i],
                                        lfb=g_LFB_val, LFB_length=val_num_each[i])
        long_feature = np.array(long_feature)
        long_feature = np.squeeze(long_feature, axis=0)
        long_feature = long_feature.transpose(-1, -2)
        path_whole = val_list_of_examples[i]
        np.save(npy_path + path_whole[:-4] + '.npy', long_feature)
        print(path_whole)
        print(long_feature.shape)

if __name__ == "__main__":
    print('number of gpu   : {:6d}'.format(num_gpu))
    print('valid batch size: {:6d}'.format(val_batch_size))
    print('num of workers  : {:6d}'.format(workers))

    train_start_vidx = []
    count = 0
    train_num_each = []
    with open(train_vid_list_file, 'r') as csv_file:
        reader = csv.DictReader(csv_file)  # First row is treated as column headers
        train_list_of_examples = [row['Video_name'] for row in reader if 'Video_name' in row and row['Video_name']]
    for name in train_list_of_examples:
        folder_path = os.path.join(image_root_path, name[0:5] + '-' + name[5:7])
        label_path = os.path.join(label_dir, name)
        files = os.listdir(folder_path)
        image_files = natsorted(files)
        file_ptr = open(label_path, 'r')
        labels = file_ptr.read().split('\n')[:-1]  # read ground truth
        min_count = min(len(image_files), len(labels))
        train_start_vidx.append(count)
        train_num_each.append(min_count)
        count += min_count

    val_start_vidx = []
    count = 0
    val_num_each = []
    with open(val_vid_list_file, 'r') as csv_file:
        reader = csv.DictReader(csv_file)  # First row is treated as column headers
        val_list_of_examples = [row['Video_name'] for row in reader if 'Video_name' in row and row['Video_name']]
    for name in val_list_of_examples:
        folder_path = os.path.join(image_root_path, name[0:5] + '-' + name[5:7])
        label_path = os.path.join(label_dir, name)
        files = os.listdir(folder_path)
        image_files = natsorted(files)
        file_ptr = open(label_path, 'r')
        labels = file_ptr.read().split('\n')[:-1]  # read ground truth
        min_count = min(len(image_files), len(labels))
        val_start_vidx.append(count)
        val_num_each.append(min_count)
        count += min_count


    train_dataset, val_dataset = get_data(image_root_path, label_dir, train_vid_list_file, val_vid_list_file, actions_dict)

    assert len(train_dataset) == (train_start_vidx[-1] + train_num_each[-1]), "Number of image folders and feature must match!"
    assert len(val_dataset) == (val_start_vidx[-1] + val_num_each[-1]), "Number of image folders and feature must match!"

    g_LFB_train, g_LFB_val = generate_LFB(train_dataset, val_dataset)
    generate_npy(g_LFB_train, g_LFB_val, train_list_of_examples, val_list_of_examples, train_num_each, val_num_each)

    print('Done')
    print()
