import torchvision 
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., thresh=0.2):
        self.mean = mean
        self.std = std
        self.thresh = thresh
        
    def __call__(self, tensor):
        noise = torch.zeros_like(tensor)
        noise[tensor>self.thresh] = 1
        noise *= torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise
    
    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'


class TextProcessor:
    def __init__(self, alphabet):
        self.alphabet = alphabet
        self.pad_token = "[PAD]"
        self.stoi = {s: i for i, s in enumerate(self.alphabet,1)}
        self.stoi[self.pad_token] = 0
        self.itos = {i: s for s, i in self.stoi.items()}
        
    def encode(self, label):
        return [self.stoi[s] for s in label]
    
    def decode(self, ids):
        return ''.join([self.itos[i] for i in ids])
    
    def __len__(self):
        return len(self.alphabet) + 1

transform_train = torchvision.transforms.Compose(
    [   
        torchvision.transforms.Grayscale(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomApply([
            torchvision.transforms.RandomAdjustSharpness(sharpness_factor=80),
            AddGaussianNoise(mean=0, std=1),
            ])
    ]
)

transform_eval = torchvision.transforms.Compose(
    [   
        torchvision.transforms.Grayscale(),
        torchvision.transforms.ToTensor()
    ]
)

class CRNNDataset(Dataset):
    def __init__(
            self, 
            height, 
            text_processor:TextProcessor, 
            transforms:torchvision.transforms,
            dataset=None
            ) -> None:
        super().__init__()

        self.height = height
        self.transform = transforms
        self.dataset = dataset
        
        self.text_processor = text_processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        dset = self.dataset[idx]
        image, text = dset['image'], dset['text']
        label = torch.tensor(self.text_processor.encode(text), dtype=torch.long)
        original_width, original_height = image.size
        new_width = int(self.height * original_width / original_height)  # Calculate width to preserve aspect ratio
        image = image.resize((new_width, self.height))
        image = self.transform(image)
        return image, label
    

def collate_fn(batch):
    images, labels = zip(*batch)
    
    max_h = max(img.size(1) for img in images)
    max_w = max(img.size(2) for img in images)
    
    padded_images = []

    for img in images:
        h, w = img.size(1), img.size(2)
        padding = (0, max_w - w, 0, max_h - h)  # left, right, top, bottom
        padded_img = torch.nn.functional.pad(img, padding, mode='constant', value=0)
        padded_images.append(padded_img)
    
    images = torch.stack(padded_images, 0)
    
    target_lengths = torch.tensor([len(label) for label in labels]).long()

    labels = pad_sequence(labels, batch_first=True, padding_value=0)
    
    return images, labels, target_lengths