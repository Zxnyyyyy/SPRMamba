import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import time
import argparse
import copy
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics
from optimizer import build_optimizer
from lr_scheduler import build_scheduler
import resnet
import logging
import sys
from dataset_Cholec80 import ImageFolderDataset
import numpy as np
import random

parser = argparse.ArgumentParser(description='lstm training')
parser.add_argument('--gpu', default=0, type=int, help='which gpu to use, default 0')
parser.add_argument('--train', default=160, type=int, help='train batch size, default 400')
parser.add_argument('--val', default=80, type=int, help='valid batch size, default 10')
parser.add_argument('--epo', default=200, type=int, help='epochs to train and val, default 25')
parser.add_argument('--work', default=4, type=int, help='num of workers to use, default 4')
parser.add_argument('--optimizer', default='adamw', type=str, help='which optimizer to use, default adamw')
parser.add_argument('--lr_scheduler', default='cosine', type=str, help='which lr_scheduler to use, default cosine')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate for optimizer, default 5e-5')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for sgd, default 0.9')
parser.add_argument('--weightdecay', default=0.05, type=float, help='weight decay for sgd, default 0')

args = parser.parse_args()

gpu_usg = args.gpu
train_batch_size = args.train
val_batch_size = args.val
epochs = args.epo
workers = args.work
learning_rate = args.lr
momentum = args.momentum
weight_decay = args.weightdecay
optimizer_name = args.optimizer
lr_scheduler_name = args.lr_scheduler
num_gpu = torch.cuda.device_count()
use_gpu = torch.cuda.is_available()
device = torch.device("cuda:{}".format(gpu_usg) if use_gpu else "cpu")

num_classes = 7
mean = [0.40774766, 0.25766712, 0.25444105]
std = [0.22692588, 0.20076735, 0.19523472]
it = 50

seed = random.randint(0, 2 ** 32 - 1)
# seed = 12345698
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

log_format = "%(asctime)s %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt="%m/%d %I:%M:%S :")
log_path = './Cholec80/log/ResNet50_' + str(seed) + '/'
pth_path = './Cholec80/pth/ResNet50_' + str(seed) + '/'
if not os.path.exists(log_path):
    os.makedirs(log_path)
fh = logging.FileHandler(os.path.join(log_path, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

image_root_path = './datasets/Cholec80/picture/'
label_dir = './datasets/Cholec80/groundTruth/'
train_vid_list_file = './datasets/Cholec80/splits_Diffusion/train.csv'
val_vid_list_file = './datasets/Cholec80/splits_Diffusion/test.csv'
mapping_file = "./datasets/Cholec80/mapping.txt"

file_ptr = open(mapping_file, 'r')
actions = file_ptr.read().split('\n')[:-1]
file_ptr.close()
actions_dict = dict()
for a in actions:
    actions_dict[a.split()[1]] = int(a.split()[0])

def get_data(image_root_path, label_dir, train_vid_list_file, val_vid_list_file, actions_dict):

    train_transforms = transforms.Compose([
        transforms.Resize((250, 250)),
        transforms.RandomCrop(224),
        transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    train_dataset = ImageFolderDataset(train_vid_list_file, image_root_path, label_dir, actions_dict, train_transforms)
    val_dataset = ImageFolderDataset(val_vid_list_file, image_root_path, label_dir, actions_dict, test_transforms)

    return train_dataset, val_dataset

def train_model(train_dataset, val_dataset):
    # TensorBoard
    writer = SummaryWriter(log_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=False
    )


    model = resnet.resnet50(num_class=num_classes)

    model.to(device)
    optimizer = build_optimizer(args, model)
    start_epoch = -1

    exp_lr_scheduler = build_scheduler(args, optimizer, len(train_loader))
    criterion_phase = nn.CrossEntropyLoss(reduction='sum')

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_accuracy_phase = 0.0
    correspond_train_acc_phase = 0.0
    best_epoch = 0

    for epoch in range(start_epoch+1, epochs):
        # Sets the module in training mode.
        model.train()
        train_loss_phase = 0.0
        train_corrects_phase = 0
        batch_progress = 0.0
        running_loss_phase = 0.0
        minibatch_correct_phase = 0.0
        train_start_time = time.time()

        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            if use_gpu:
                inputs, labels_phase = data[0].to(device), data[1].to(device)
            else:
                inputs, labels_phase = data[0], data[1]
            labels_phase = labels_phase.to(torch.long)

            outputs_phase = model.forward(inputs)
            _, preds_phase = torch.max(outputs_phase.data, 1)
            loss_phase = criterion_phase(outputs_phase, labels_phase)

            loss_phase.backward()
            optimizer.step()
            exp_lr_scheduler.step(epoch * len(train_loader) + i)

            running_loss_phase += loss_phase.data.item()
            train_loss_phase += loss_phase.data.item()

            batch_corrects_phase = torch.sum(preds_phase == labels_phase.data)
            train_corrects_phase += batch_corrects_phase
            minibatch_correct_phase += batch_corrects_phase

            if i % it == it - 1:
                # ...log the running loss
                batch_iters = epoch * len(train_loader) + i
                writer.add_scalar('training loss phase',
                                  running_loss_phase / (train_batch_size * it),
                                  batch_iters)
                # ...log the training acc
                writer.add_scalar('training acc phase',
                                  float(minibatch_correct_phase) / (float(train_batch_size) * it),
                                  batch_iters)

                running_loss_phase = 0.0
                minibatch_correct_phase = 0.0

            if (i + 1) >= len(train_loader):
                running_loss_phase = 0.0
                minibatch_correct_phase = 0.0

            batch_progress += 1
            if batch_progress >= len(train_loader):
                percent = 100.0
                print('Batch progress: %s [%d/%d]' % (str(percent) + '%', len(train_loader), len(train_loader)), end='\n')
            else:
                percent = round(batch_progress / len(train_loader) * 100, 2)
                print('Batch progress: %s [%d/%d]' % (
                str(percent) + '%', batch_progress , len(train_loader)), end='\r')

        train_elapsed_time = time.time() - train_start_time
        train_accuracy_phase = float(train_corrects_phase) / float(len(train_loader) * train_batch_size)
        train_average_loss_phase = train_loss_phase / (len(train_loader) * train_batch_size)

        # Sets the module in evaluation mode.
        model.eval()
        val_corrects_phase = 0
        val_start_time = time.time()
        val_progress = 0
        val_all_preds_phase = []
        val_all_labels_phase = []

        with torch.no_grad():
            for data in val_loader:
                if use_gpu:
                    inputs, labels_phase = data[0].to(device), data[1].to(device)
                else:
                    inputs, labels_phase = data[0], data[1]
                labels_phase = labels_phase.to(torch.long)

                outputs_phase = model.forward(inputs)
                _, preds_phase = torch.max(outputs_phase.data, 1)
                val_corrects_phase += torch.sum(preds_phase == labels_phase.data)

                for i in range(len(preds_phase)):
                    val_all_preds_phase.append(int(preds_phase.data.cpu()[i]))
                for i in range(len(labels_phase)):
                    val_all_labels_phase.append(int(labels_phase.data.cpu()[i]))

                val_progress += 1
                if val_progress >= len(val_loader):
                    percent = 100.0
                    print('Val progress: %s [%d/%d]' % (str(percent) + '%', len(val_loader), len(val_loader)), end='\n')
                else:
                    percent = round(val_progress / len(val_loader) * 100, 2)
                    print('Val progress: %s [%d/%d]' % (str(percent) + '%', val_progress, len(val_loader)),
                          end='\r')

        val_elapsed_time = time.time() - val_start_time
        val_accuracy_phase = float(val_corrects_phase) / float(len(val_loader) * val_batch_size)

        val_recall_phase = metrics.recall_score(val_all_labels_phase, val_all_preds_phase, average='macro')
        val_precision_phase = metrics.precision_score(val_all_labels_phase, val_all_preds_phase, average='macro')
        val_jaccard_phase = metrics.jaccard_score(val_all_labels_phase, val_all_preds_phase, average='macro')
        val_precision_each_phase = metrics.precision_score(val_all_labels_phase, val_all_preds_phase, average=None)
        val_precision_each_phase = [f"{num:.4f}" for num in val_precision_each_phase]
        val_precision_each_phase = list(map(float, val_precision_each_phase))
        val_recall_each_phase = metrics.recall_score(val_all_labels_phase, val_all_preds_phase, average=None)
        val_recall_each_phase = [f"{num:.4f}" for num in val_recall_each_phase]
        val_recall_each_phase = list(map(float, val_recall_each_phase))

        writer.add_scalar('validation acc epoch phase',
                          float(val_accuracy_phase), epoch)

        logging.info(f'epoch            : {epoch}')
        logging.info(f'train in         : {train_elapsed_time // 60:2.0f}m{train_elapsed_time % 60:2.0f}s')
        logging.info(f'train loss(phase): {train_average_loss_phase:4.4f}')
        logging.info(f'train accu(phase): {train_accuracy_phase:.4f}')
        logging.info(f'valid in         : {val_elapsed_time // 60:2.0f}m{val_elapsed_time % 60:2.0f}s')
        logging.info(f'valid accu(phase): {val_accuracy_phase:.4f}')

        logging.info(f"val_precision_each_phase:{val_precision_each_phase}")
        logging.info(f"val_recall_each_phase   :{val_recall_each_phase}")
        logging.info(f"val_precision_phase     :{val_precision_phase:.4f}")
        logging.info(f"val_recall_phase        :{val_recall_phase:.4f}")
        logging.info(f"val_jaccard_phase       :{val_jaccard_phase:.4f}")


        if val_accuracy_phase > best_val_accuracy_phase:
            best_val_accuracy_phase = val_accuracy_phase
            correspond_train_acc_phase = train_accuracy_phase
            best_model_wts = copy.deepcopy(model.state_dict())
            best_epoch = epoch

            # Best stats
            best_tr_loss = train_average_loss_phase
            best_tr_acc = train_accuracy_phase
            best_v_acc = val_accuracy_phase
            best_v_pr = val_precision_phase
            best_v_re = val_recall_phase
            best_v_ji = val_jaccard_phase
            best_v_pr_each = val_precision_each_phase
            best_v_re_each = val_recall_each_phase

        if val_accuracy_phase == best_val_accuracy_phase:
            if train_accuracy_phase > correspond_train_acc_phase:
                correspond_train_acc_phase = train_accuracy_phase
                best_model_wts = copy.deepcopy(model.state_dict())
                best_epoch = epoch

                # Best stats
                best_tr_loss = train_average_loss_phase
                best_tr_acc = train_accuracy_phase
                best_v_acc = val_accuracy_phase
                best_v_pr = val_precision_phase
                best_v_re = val_recall_phase
                best_v_ji = val_jaccard_phase
                best_v_pr_each = val_precision_each_phase
                best_v_re_each = val_recall_each_phase

        save_val_phase = int("{:4.0f}".format(best_val_accuracy_phase * 10000))
        save_train_phase = int("{:4.0f}".format(correspond_train_acc_phase * 10000))
        save_path = os.path.join(pth_path, "best_model")
        save_path1 = os.path.join(pth_path, "temp")
        base_name = "ResNet50_epoch_" + str(best_epoch) \
                    + str(optimizer_name) \
                    + str(lr_scheduler_name) \
                    + "_batch_" + str(train_batch_size) \
                    + "_train_" + str(save_train_phase) \
                    + "_val_" + str(save_val_phase)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(save_path1):
            os.makedirs(save_path1)

        torch.save(best_model_wts, os.path.join(save_path, str(base_name) + ".pth"))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_phase.item(),
        }, os.path.join(save_path1, str(epoch) + ".pth"))

        logging.info(f"best_epoch              :{str(best_epoch)}")
        logging.info(f"best stats:")
        logging.info(f"best_train_loss         :{best_tr_loss:.4f}")
        logging.info(f"best_train_acc          :{best_tr_acc:.4f}")
        logging.info(f"best_val_acc            :{best_v_acc:.4f}")
        logging.info(f"best_val_precision      :{best_v_pr:.4f}")
        logging.info(f"best_val_recall         :{best_v_re:.4f}")
        logging.info(f"best_val_jaccard        :{best_v_ji:.4f}")
        logging.info(f"best_val_precision_each :{best_v_pr_each}")
        logging.info(f"best_val_recall_each    :{best_v_re_each}")


def main():
    train_dataset, val_dataset = get_data(image_root_path, label_dir, train_vid_list_file, val_vid_list_file, actions_dict)
    train_model(train_dataset, val_dataset)

if __name__ == "__main__":
    logging.info(f'number of gpu   : {num_gpu}')
    logging.info(f'num_classes     : {num_classes}')
    logging.info(f'train batch size: {train_batch_size}')
    logging.info(f'valid batch size: {val_batch_size}')
    logging.info(f'optimizer choice: {optimizer_name}')
    logging.info(f'lr choice       : {lr_scheduler_name}')
    logging.info(f'num of epochs   : {epochs}')
    logging.info(f'num of workers  : {workers}')
    logging.info(f'learning rate   : {learning_rate}')
    logging.info(f'momentum for sgd: {momentum}')
    logging.info(f'weight decay    : {weight_decay}')

    main()

    logging.info('Done')
    print()