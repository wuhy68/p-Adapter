import os
import json
import random
from PIL import Image

from data.utils import pre_question

import torch
from torch.utils.data import Dataset

class snli_ve_dataset(Dataset):
    def __init__(self, transform, image_root, ann_root, split="train"):

        filenames = {'train': 'snli_ve_train.json', 'val': 'snli_ve_dev.json', 'test': 'snli_ve_test.json'}

        self.annotation = json.load(open(os.path.join(ann_root, filenames[split]),'r'))
        self.transform = transform
        self.image_root = image_root

        self.answer_list = ['Correct', 'Incorrect', 'Ambiguous']

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.image_root, ann['img'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        if ann['hypothesis'][-1] == '.':
            ann['hypothesis'] = ann['hypothesis'][:-1]

        hypothesis = 'Is it correct that ' + ann['hypothesis'].lower() + ' ? '
        label = ann['label_s']

        return image, hypothesis, label