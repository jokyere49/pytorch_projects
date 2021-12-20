from PIL import Image
import os
import csv
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']


class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path):

        self.dataset_path = dataset_path
        self.labelsFileContent = []
        labels = os.path.join(dataset_path, 'labels.csv')
        with open(labels, newline='') as csvfile:
            labelReader = csv.reader(csvfile, delimiter=',')
            for row in labelReader:
                self.labelsFileContent.append(row)
        self.labelsFileContent = self.labelsFileContent[1:]

    def __len__(self):

        return len(self.labelsFileContent)

    def __getitem__(self, idx):

        # The label processing
        labelDictionary = {}
        for labelNumber, labelName in enumerate(LABEL_NAMES):
            labelDictionary[labelName] = labelNumber
        labelString = self.labelsFileContent[idx][1]
        label = labelDictionary[labelString]

        # Image processing
        img_path = os.path.join(self.dataset_path, self.labelsFileContent[idx][0])
        image_to_tensor = transforms.ToTensor()
        image = image_to_tensor(Image.open(img_path))

        return image, label


def load_data(dataset_path, num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()
