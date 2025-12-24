import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import csv
from torchvision import transforms

class ImageFolderDataset(Dataset):
    def __init__(self, vid_list_file, image_root_dir, label_dir, actions_dict, transform=None):
        """
        Args:
            image_root_dir (str): Path to the directory containing folders of images (one folder per video).
            label_dir (str): Path to the directory containing label files (one file per video).
            transform (callable, optional): Optional transform to be applied to a sample.
        """
        self.image_root_dir = image_root_dir
        self.label_dir = label_dir
        self.transform = transform
        self.actions_dict = actions_dict

        with open(vid_list_file, 'r') as csv_file:
            reader = csv.DictReader(csv_file)  # First row is treated as column headers
            self.list_of_examples = [row['Video_name'] for row in reader if 'Video_name' in row and row['Video_name']]

        # Create a list of all images and their corresponding labels
        self.image_label_pairs = []
        for name in self.list_of_examples:
            folder_path = os.path.join(self.image_root_dir, name[:-4])
            # folder_path = os.path.join(self.image_root_dir, name[0:5] + '-' + name[5:7])
            label_path = os.path.join(self.label_dir, name)

            image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))])
            file_ptr = open(label_path, 'r')
            labels = file_ptr.read().split('\n')[:-1]  # read ground truth

            # Match the number of images and labels to the minimum of the two
            min_count = min(len(image_files), len(labels))
            image_files = image_files[:min_count]
            labels = labels[:min_count]
            for i in range(min_count):
                labels[i] = self.actions_dict[labels[i]]
            for img_file, label in zip(image_files, labels):
                self.image_label_pairs.append((os.path.join(folder_path, img_file), float(label)))

    def __len__(self):
        return len(self.image_label_pairs)

    def __getitem__(self, idx):
        image_path, label = self.image_label_pairs[idx]

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        return image, label


# Usage example
if __name__ == "__main__":
    vid_list_file = './datasets/splits/total.csv'
    image_root_dir = "./datasets/picture/"
    label_dir = "./datasets/groundTruth/"
    mapping_file = './datasets/mapping.txt'

    file_ptr = open(mapping_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])

    mean = torch.zeros(3)
    std = torch.zeros(3)

    # Define a dataset
    dataset = ImageFolderDataset(vid_list_file, image_root_dir, label_dir, actions_dict,
                                 transform=transforms.ToTensor())

    # Define a DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    for X, _ in dataloader:
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()

    mean.div_(len(dataloader))
    std.div_(len(dataloader))

    print(mean.numpy())
    print(std.numpy())
