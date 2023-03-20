from torch.utils import data


class UCR_Dataset(data.Dataset):

    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        return tuple((self.data[index],self.target[index]))