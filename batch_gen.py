# Only used in MS-TCN model
import torch
import numpy as np
import random
import csv
class BatchGenerator(object):
    def __init__(self, num_classes, actions_dict, gt_path, features_path):
        self.list_of_examples = list()
        self.index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.gt_path = gt_path
        self.features_path = features_path

    def reset(self):
        self.index = 0

    def reset_shuffle(self):
        self.index = 0
        random.shuffle(self.list_of_examples)

    def has_next(self):
        if self.index < len(self.list_of_examples):
            return True
        return False

    def read_data(self, vid_list_file):
        '''
        read data and random shuffle the examples
        :param vid_list_file: file name, str
        :return:
        '''
        with open(vid_list_file, 'r') as csv_file:
            reader = csv.DictReader(csv_file)  # First row is treated as column headers
            self.list_of_examples = [row['Video_name'] for row in reader if 'Video_name' in row and row['Video_name']]
        random.shuffle(self.list_of_examples)

    def next_batch(self, batch_size):
        '''
        sample next batch
        :param batch_size: int
        :return: mask[batch_size, num_classes, max(length_of_sequences)]
        '''
        batch = self.list_of_examples[self.index:self.index + batch_size]
        self.index += batch_size  # use index to get random sample

        batch_input = []  # feature vectors
        batch_target = []  # ground truth vector
        batch_vid = []
        for vid in batch:
            features = np.load(self.features_path + vid[:-4] + '.npy')
            file_ptr = open(self.gt_path + vid, 'r')
            content = file_ptr.read().split('\n')[:-1]  # read ground truth
            # initialize and produce gt vector
            classes = np.zeros(min(np.shape(features)[1], len(content)))
            for i in range(len(classes)):
                classes[i] = self.actions_dict[content[i]]

            batch_input.append(features)
            batch_target.append(classes)
            batch_vid.append(vid)
        length_of_sequences = list(map(len, batch_target))  # get length of batch_target
        # create pytorch tensor
        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max(length_of_sequences), dtype=torch.float)
        batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long)*(-100)
        mask = torch.zeros(len(batch_input), self.num_classes, max(length_of_sequences), dtype=torch.float)
        mask.require_grad = False
        for i in range(len(batch_input)):
            batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
            batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
            # actually np.shape(batch_target[i])[0]=np.shape(batch_input[i])[1], =this sequence's length
            # mask: record this sequence's length, total=max(length_of_sequences)
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])
        return batch_input_tensor, batch_target_tensor, mask, batch_vid
