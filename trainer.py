import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import copy
import os
import csv
import sys
from optimizer import MultiStepLR, Adam, SequentialLR
from eval import evaluate_phase
sys.path.append(os.path.curdir)

class Trainer:
    def __init__(self):
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)  # for ms_tcn
        self.maskCE = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')  # for cascade stages
        self.nll = nn.NLLLoss(ignore_index=-100, reduction='none')   # for fusion stage
        self.mse = nn.MSELoss(reduction='none')

    def train(self, writer, model, save_dir, train_batch_gen, val_batch_gen, test_batch_gen, num_epochs, batch_size, learning_rate, device, weight_decay, num_class):
        model.to(device)
        optimizer = Adam(model, learning_rate, weight_decay)
        scheduler = SequentialLR(optimizer)
        save_best = os.path.join(save_dir, 'best_model')
        save_temp = os.path.join(save_dir, 'temp')
        if not os.path.exists(save_best):
            os.makedirs(save_best)
        if os.path.exists(save_temp):
            files = os.listdir(save_temp)
            files = [file for file in files if os.path.isfile(os.path.join(save_temp, file))]
            files.sort(key=lambda x: os.path.getmtime(os.path.join(save_temp, x)))
            saved_state = torch.load(os.path.join(save_temp, files[-1]), map_location=device)
            model.load_state_dict(saved_state['model'])
            optimizer.load_state_dict(saved_state['optimizer'])
            restore_epoch = saved_state['epoch']
        else:
            os.makedirs(save_temp)
            restore_epoch = -1
        best_model_wts = copy.deepcopy(model.state_dict())
        best_val_jaccard_phase = 0.0
        correspond_train_acc_phase = 0.0
        correspond_test_jaccard_phase = 0.0
        best_epoch = 0
        for epoch in range(restore_epoch+1, num_epochs):
            model.train()
            epoch_loss = 0
            correct = 0
            total = 0
            while train_batch_gen.has_next():
                batch_input, batch_target, mask, vid = train_batch_gen.next_batch(batch_size)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                optimizer.zero_grad()
                predictions = model(batch_input)
                loss = 0
                for p in predictions:
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, num_class), batch_target.view(-1))
                    loss += 0.15*torch.mean(torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16))

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
                scheduler.step()

                _, predicted = torch.max(predictions[-1].data, 1)
                correct += ((predicted == batch_target).float()*mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()
            train_accuracy_phase = float(correct) / total

            print("[epoch %d]: epoch loss = %f,   acc = %f,    lr=%f" % (
            epoch, epoch_loss / len(train_batch_gen.list_of_examples), float(correct) / total,
            optimizer.param_groups[0]['lr']))
            writer.add_scalar('Train_loss', epoch_loss / len(train_batch_gen.list_of_examples), epoch)
            writer.add_scalar('Train_Acc', train_accuracy_phase, epoch)
            train_batch_gen.reset_shuffle()

            model.eval()
            val_all_preds_phase = []
            val_all_labels_phase = []
            val_acc_each_video = []
            while val_batch_gen.has_next():
                batch_input, batch_target, mask, vid = val_batch_gen.next_batch(batch_size)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                predictions = model(batch_input)
                _, predicted = torch.max(predictions[-1].data, 1)
                predicted = predicted.squeeze()
                batch_target = batch_target.squeeze()
                val_acc_each_video.append(float(torch.sum(predicted == batch_target.data)) / np.shape(batch_input)[2])

                for j in range(len(predicted)):
                    val_all_preds_phase.append(int(predicted.data.cpu()[j]))
                for j in range(len(batch_target)):
                    val_all_labels_phase.append(int(batch_target.data.cpu()[j]))

            val_acc_video = np.mean(val_acc_each_video)
            val_recall_phase, val_precision_phase, val_jaccard_phase = evaluate_phase(val_all_preds_phase, val_all_labels_phase)

            writer.add_scalar('Val_Acc', val_acc_video, epoch)
            writer.add_scalar('Val_Recall', val_recall_phase, epoch)
            writer.add_scalar('Val_Precision', val_precision_phase, epoch)
            writer.add_scalar('Val_Jaccard', val_jaccard_phase, epoch)
            val_batch_gen.reset()

            model.eval()
            test_all_preds_phase = []
            test_all_labels_phase = []
            test_acc_each_video = []
            while test_batch_gen.has_next():
                batch_input, batch_target, mask, vid = test_batch_gen.next_batch(batch_size)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                predictions = model(batch_input)
                _, predicted = torch.max(predictions[-1].data, 1)
                predicted = predicted.squeeze()
                batch_target = batch_target.squeeze()
                test_acc_each_video.append(float(torch.sum(predicted == batch_target.data)) / np.shape(batch_input)[2])

                for j in range(len(predicted)):
                    test_all_preds_phase.append(int(predicted.data.cpu()[j]))
                for j in range(len(batch_target)):
                    test_all_labels_phase.append(int(batch_target.data.cpu()[j]))

            test_acc_video = np.mean(test_acc_each_video)
            test_recall_phase, test_precision_phase, test_jaccard_phase = evaluate_phase(test_all_preds_phase, test_all_labels_phase)

            writer.add_scalar('Test_Acc', test_acc_video, epoch)
            writer.add_scalar('Test_Recall', test_recall_phase, epoch)
            writer.add_scalar('Test_Precision', test_precision_phase, epoch)
            writer.add_scalar('Test_Jaccard', test_jaccard_phase, epoch)
            test_batch_gen.reset()

            if val_jaccard_phase > best_val_jaccard_phase:
                best_val_jaccard_phase = val_jaccard_phase
                correspond_test_jaccard_phase = test_jaccard_phase
                correspond_train_acc_phase = train_accuracy_phase
                best_model_wts = copy.deepcopy(model.state_dict())
                best_epoch = epoch

            if val_jaccard_phase == best_val_jaccard_phase:
                if train_accuracy_phase > correspond_train_acc_phase:
                    correspond_train_acc_phase = train_accuracy_phase
                    correspond_test_jaccard_phase = test_jaccard_phase
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_epoch = epoch

            # save model
            save_val_phase = int("{:4.0f}".format(best_val_jaccard_phase * 10000))
            save_test_phase = int("{:4.0f}".format(correspond_test_jaccard_phase * 10000))
            save_train_phase = int("{:4.0f}".format(correspond_train_acc_phase * 10000))

            base_name = "BCN_epoch_" + str(best_epoch) \
                        + "_train_" + str(save_train_phase) \
                        + "_val_" + str(save_val_phase) \
                        + "_test_" + str(save_test_phase)
            torch.save(best_model_wts, os.path.join(save_best, str(base_name) + ".pth"))
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}, os.path.join(save_temp, str(epoch) + ".pth"))

    def predict(self, model, model_path, results_dir, actions_dict, device, test_batch_gen, batch_size):
        model.eval()
        inverse_dict = {v: k for k, v in actions_dict.items()}
        model.to(device)
        model.load_state_dict(torch.load(model_path)['model'])
        with torch.no_grad():
            begin_time = time.time()
            while test_batch_gen.has_next():
                batch_input, batch_target, mask, vid = test_batch_gen.next_batch(batch_size)
                batch_target = batch_target.squeeze(0)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                predictions = model(batch_input)
                predictions = predictions[-1]
                _, predicted = torch.max(predictions.data, 1)
                predicted = predicted.squeeze()
                predicted = predicted.cpu().numpy()
                f = open(results_dir + "/" + vid[0], "w")
                for j in range(len(predicted)):
                    f.write(str(inverse_dict[predicted[j]]))
                    f.write('\n')
                f.close()
            end_time = time.time()
            print(end_time - begin_time)
                # for i in range(len(predictions)):
                #     predictions = F.softmax(predictions[i], dim=1)
                #     _, predicted = torch.max(predictions.data, 1)
                #     predicted = predicted.squeeze()
                #     predicted = predicted.cpu().numpy()
                #     f = open(results_dir + "/" + vid + '_stage' + str(i), "w")
                #     for j in range(len(predicted)):
                #         f.write(str(inverse_dict[predicted[j]]))
                #         f.write('\n')
                #     f.close()