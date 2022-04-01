from torch.utils.data.dataset import Dataset
import os
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        super(CustomDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.sample_file = []
        # directories = [d for d in os.listdir(root) if os.path.isdir(root+d)]
        # for d in directories:
        #     files = os.listdir(root+d)
        #     # files.sort(key=lambda x: int(x[:-4]))
        #     for f in files:
        #         item = self.root+d+os.path.sep+f, int(d)
        #         self.sample_file.append(item)
        files = os.listdir(root)
        files.sort(key=lambda x: int(x[:-4]))
        for f in files:
            item = self.root+os.path.sep+f, int(0)
            self.sample_file.append(item)

    def __getitem__(self, item: int):
        file_name, label = self.sample_file[item]
        sample = Image.open(file_name)
        if self.transform is not None:
            sample = self.transform(sample)
        # if len(sample.size()) != 3:
        #     print(sample.size())
        return sample, label

    def __len__(self):
        return len(self.sample_file)
