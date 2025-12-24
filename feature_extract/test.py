import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import resnet
from dataset import ImageFolderDataset
import csv
from natsort import natsorted
from timm.models import create_model
from sklearn import metrics
from tqdm import tqdm

parser = argparse.ArgumentParser(description='resnet50 training')
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

output_path = './ResNet50_prediction/'
image_root_path = './datasets/ESD385/picture/'
label_dir = './datasets/ESD385/groundTruth/'
test_vid_list_file = './datasets/ESD385/splits/test.csv'
mapping_file = "./datasets/ESD385/mapping.txt"

if not os.path.exists(output_path):
    os.makedirs(output_path)

file_ptr = open(mapping_file, 'r')
actions = file_ptr.read().split('\n')[:-1]
file_ptr.close()
actions_dict = dict()
for a in actions:
    actions_dict[a.split()[1]] = int(a.split()[0])
inverse_dict = {v: k for k, v in actions_dict.items()}
num_classes = 8
mean = [0.4921294, 0.34441853, 0.29559746]
std = [0.22656082, 0.2103699,  0.18450819]

def get_data(image_root_path, label_dir, test_vid_list_file, actions_dict):

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    test_dataset = ImageFolderDataset(test_vid_list_file, image_root_path, label_dir, actions_dict, test_transforms)

    return test_dataset

def evaluate_phase(all_prediction, all_labels):
    recall_phase = metrics.recall_score(all_prediction, all_labels, average='macro')
    precision_phase = metrics.precision_score(all_prediction, all_labels, average='macro')
    jaccard_phase = metrics.jaccard_score(all_prediction, all_labels, average='macro')
    return recall_phase, precision_phase, jaccard_phase

def generate_LFB(test_dataset, test_list_of_examples, test_num_each, test_start_vidx):

    test_feature_loader = DataLoader(
        test_dataset,
        batch_size=val_batch_size,
        num_workers=workers,
        shuffle=False,
        pin_memory=False
    )

    model = resnet.resnet50(num_class=num_classes)
    model_weight_path = './feature_extract/pth/temp/110.pth'
    model.load_state_dict(torch.load(model_weight_path, map_location=device)['model_state_dict'], strict=False)
    model.eval()

    if use_gpu:
        model.to(device)

    with torch.no_grad():
        test_all_preds_phase = []
        test_all_labels_phase = []
        for data in test_feature_loader:
            if use_gpu:
                inputs, labels_phase = data[0].to(device), data[1].to(device)
            else:
                inputs, labels_phase = data[0], data[1]

            outputs_phase = model.forward(inputs)
            _, preds_phase = torch.max(outputs_phase.data, 1)
            for i in range(len(preds_phase)):
                test_all_preds_phase.append(int(preds_phase.data.cpu()[i]))
            for j in range(len(labels_phase)):
                test_all_labels_phase.append(int(labels_phase.data.cpu()[j]))
    test_recall_phase, test_precision_phase, test_jaccard_phase = evaluate_phase(test_all_preds_phase,
                                                                                 test_all_labels_phase)
    print(test_recall_phase, test_precision_phase, test_jaccard_phase)
    f = open('./ResNet50.txt', "w")
    for j in range(len(test_all_preds_phase)):
        f.write(str(inverse_dict[test_all_preds_phase[j]]))
        f.write('\n')
    f.close()
    with open('./ResNet50.txt', 'r') as f:
        lines = f.readlines()  # 自动保留换行符
    for name in test_list_of_examples:
        with open(os.path.join(output_path, name), 'w') as f:
            f.writelines(lines[test_start_vidx[name]:test_start_vidx[name]+test_num_each[name]])


if __name__ == "__main__":
    print('number of gpu   : {:6d}'.format(num_gpu))
    print('valid batch size: {:6d}'.format(val_batch_size))
    print('num of workers  : {:6d}'.format(workers))

    test_start_vidx = {}
    count = 0
    test_num_each = {}
    with open(test_vid_list_file, 'r') as csv_file:
        reader = csv.DictReader(csv_file)  # First row is treated as column headers
        test_list_of_examples = [row['Video_name'] for row in reader if 'Video_name' in row and row['Video_name']]
    for name in test_list_of_examples:
        folder_path = os.path.join(image_root_path, 'M_' + name[:-4])
        label_path = os.path.join(label_dir, name)
        files = os.listdir(folder_path)
        image_files = natsorted(files)
        file_ptr = open(label_path, 'r')
        labels = file_ptr.read().split('\n')[:-1]  # read ground truth
        min_count = min(len(image_files), len(labels))
        test_start_vidx[name] = count
        test_num_each[name] = min_count
        count += min_count

    test_dataset = get_data(image_root_path, label_dir, test_vid_list_file, actions_dict)

    assert len(test_dataset) == (test_start_vidx[test_list_of_examples[-1]] + test_num_each[test_list_of_examples[-1]]), "Number of image folders and feature must match!"

    generate_LFB(test_dataset, test_list_of_examples, test_num_each, test_start_vidx)


    print('Done')
    print()
