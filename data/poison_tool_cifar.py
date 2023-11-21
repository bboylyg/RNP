from torchvision import transforms, datasets
from torchvision.datasets import CIFAR10
from torch.utils.data import random_split, DataLoader, Dataset
import torch
import numpy as np
import time
import argparse
from tqdm import tqdm
from copy import deepcopy
from PIL import Image
import torch.nn.functional as F

import sys
sys.path.append("..")
from models import dynamic_models

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


MEAN_CIFAR10 = (0.4914, 0.4822, 0.4465)
STD_CIFAR10 = (0.2023, 0.1994, 0.2010)

def split_dataset(dataset, frac=0.1, perm=None):
    """
    :param dataset: The whole dataset which will be split.
    """
    if perm is None:
        perm = np.arange(len(dataset))
        np.random.shuffle(perm)
    nb_split = int(frac * len(dataset))

    # generate the training set
    train_set = deepcopy(dataset)
    train_set.data = train_set.data[perm[nb_split:]]
    train_set.targets = np.array(train_set.targets)[perm[nb_split:]].tolist()

    # generate the test set
    split_set = deepcopy(dataset)
    split_set.data = split_set.data[perm[:nb_split]]
    split_set.targets = np.array(split_set.targets)[perm[:nb_split]].tolist()

    print('total data size: %d images, split test size: %d images, split ratio: %f' % (
    len(train_set.targets), len(split_set.targets), frac))

    return train_set, split_set

def get_train_loader(args):
    print('==> Preparing train data..')
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        # transforms.RandomRotation(3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
    ])

    if (args.dataset == 'CIFAR10'):
        trainset = datasets.CIFAR10(root='data/CIFAR10', train=True, download=True)
    else:
        raise Exception('Invalid dataset')

    train_data = DatasetCL(args, full_dataset=trainset, transform=tf_train)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    return train_loader

def get_test_loader(args):
    print('==> Preparing test data..')
    tf_test = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
                                  ])
    if (args.dataset == 'CIFAR10'):
        testset = datasets.CIFAR10(root='data/CIFAR10', train=False, download=True)
    else:
        raise Exception('Invalid dataset')

    test_data_clean = DatasetBD(args, full_dataset=testset, inject_portion=0, transform=tf_test, mode='test')
    test_data_bad = DatasetBD(args, full_dataset=testset, inject_portion=1, transform=tf_test, mode='test')

    # (apart from label 0) bad test data
    test_clean_loader = DataLoader(dataset=test_data_clean,
                                       batch_size=args.batch_size,
                                       shuffle=False,
                                       )
    # all clean test data
    test_bad_loader = DataLoader(dataset=test_data_bad,
                                       batch_size=args.batch_size,
                                       shuffle=False,
                                       )

    return test_clean_loader, test_bad_loader


def get_backdoor_loader(args):
    print('==> Preparing train data..')
    tf_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        # transforms.RandomRotation(3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
                                  ])
    if (args.dataset == 'CIFAR10'):
        trainset = datasets.CIFAR10(root='data/CIFAR10', train=True, download=True)
    else:
        raise Exception('Invalid dataset')

    train_data_bad = DatasetBD(args, full_dataset=trainset, inject_portion=args.inject_portion, transform=tf_train, mode='train')
    train_bad_loader = DataLoader(dataset=train_data_bad,
                                       batch_size=args.batch_size,
                                       shuffle=False,
                                       )

    return train_data_bad, train_bad_loader


class Dataset_npy(torch.utils.data.Dataset):
    def __init__(self, full_dataset=None, transform=None):
        self.dataset = full_dataset
        self.transform = transform
        self.dataLen = len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index][0]
        label = self.dataset[index][1]

        if self.transform:
            image = self.transform(image)
        # print(type(image), image.shape)
        return image, label

    def __len__(self):
        return self.dataLen



class DatasetCL(Dataset):
    def __init__(self, args, full_dataset=None, transform=None):
        self.dataset = self.random_split(full_dataset=full_dataset, ratio=args.ratio)
        self.transform = transform
        self.dataLen = len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index][0]
        label = self.dataset[index][1]

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return self.dataLen

    def random_split(self, full_dataset, ratio):
        print('full_train:', len(full_dataset))
        train_size = int(ratio * len(full_dataset))
        drop_size = len(full_dataset) - train_size
        train_dataset, drop_dataset = random_split(full_dataset, [train_size, drop_size])
        print('train_size:', len(train_dataset), 'drop_size:', len(drop_dataset))

        return train_dataset


def create_bd(netG, netM, inputs):
    patterns = netG(inputs)
    masks_output = netM.threshold(netM(inputs))
    return patterns, masks_output

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

class DatasetBD(Dataset):
    def __init__(self, args, full_dataset, inject_portion, transform=None, mode="train", device=torch.device("cuda"), distance=1):
        self.dataset = self.addTrigger(full_dataset, args.target_label, inject_portion, mode, distance, args.trig_w, args.trig_h, args.trigger_type, args.target_type)
        self.device = device
        self.transform = transform

    def __getitem__(self, item):
        img = self.dataset[item][0]
        label = self.dataset[item][1]
        img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.dataset)

    def addTrigger(self, dataset, target_label, inject_portion, mode, distance, trig_w, trig_h, trigger_type, target_type):
        print("Generating " + mode + "bad Imgs")
        perm = np.random.permutation(len(dataset))[0: int(len(dataset) * inject_portion)]
        # dataset
        dataset_ = list()

        cnt = 0
        for i in tqdm(range(len(dataset))):
            data = dataset[i]

            if target_type == 'all2one':

                if mode == 'train':
                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        # select trigger
                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, mode, trigger_type)

                        # change target
                        dataset_.append((img, target_label))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

                else:
                    if data[1] == target_label:
                        continue

                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, mode, trigger_type)

                        dataset_.append((img, target_label))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

            # all2all attack
            elif target_type == 'all2all':

                if mode == 'train':
                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:

                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, mode, trigger_type)
                        target_ = self._change_label_next(data[1])

                        dataset_.append((img, target_))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

                else:

                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, mode, trigger_type)

                        target_ = self._change_label_next(data[1])
                        dataset_.append((img, target_))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

            # clean label attack
            elif target_type == 'cleanLabel':

                if mode == 'train':
                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]

                    if i in perm:
                        if data[1] == target_label:

                            img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, mode, trigger_type)

                            dataset_.append((img, data[1]))
                            cnt += 1

                        else:
                            dataset_.append((img, data[1]))
                    else:
                        dataset_.append((img, data[1]))

                else:
                    if data[1] == target_label:
                        continue

                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, mode, trigger_type)

                        dataset_.append((img, target_label))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

        time.sleep(0.01)
        print("Injecting Over: " + str(cnt) + "Bad Imgs, " + str(len(dataset) - cnt) + "Clean Imgs")


        return dataset_


    def _change_label_next(self, label):
        label_new = ((label + 1) % 10)
        return label_new

    def selectTrigger(self, img, width, height, distance, trig_w, trig_h, mode, triggerType):

        assert triggerType in ['squareTrigger', 'gridTrigger', 'fourCornerTrigger', 'randomPixelTrigger',
                               'signalTrigger', 'trojanTrigger', 'CLTrigger', 'dynamicTrigger', 'nashvilleTrigger',
                               'onePixelTrigger', 'wanetTrigger']

        if triggerType == 'squareTrigger':
            img = self._squareTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'gridTrigger':
            img = self._gridTriger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'fourCornerTrigger':
            img = self._fourCornerTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'randomPixelTrigger':
            img = self._randomPixelTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'signalTrigger':
            img = self._signalTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'trojanTrigger':
            img = self._trojanTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'CLTrigger':
            img = self._CLTrigger(img, mode=mode)

        elif triggerType == 'dynamicTrigger':
            img = self._dynamicTrigger(img, mode=mode)

        elif triggerType == 'nashvilleTrigger':
            img = self._nashvilleTrigger(img, mode=mode)

        elif triggerType == 'onePixelTrigger':
            img = self._onePixelTrigger(img, mode=mode)

        elif triggerType == 'wanetTrigger':
            img = self._wanetTrigger(img, mode=mode)

        else:
            raise NotImplementedError

        return img

    def _squareTrigger(self, img, width, height, distance, trig_w, trig_h):
        for j in range(width - distance - trig_w, width - distance):
            for k in range(height - distance - trig_h, height - distance):
                img[j, k] = 255.0

        return img

    def _gridTriger(self, img, width, height, distance, trig_w, trig_h):

        img[width - 1][height - 1] = 255
        img[width - 1][height - 2] = 0
        img[width - 1][height - 3] = 255

        img[width - 2][height - 1] = 0
        img[width - 2][height - 2] = 255
        img[width - 2][height - 3] = 0

        img[width - 3][height - 1] = 255
        img[width - 3][height - 2] = 0
        img[width - 3][height - 3] = 0

        # adptive center trigger
        # alpha = 1
        # img[width - 14][height - 14] = 255* alpha
        # img[width - 14][height - 13] = 128* alpha
        # img[width - 14][height - 12] = 255* alpha
        #
        # img[width - 13][height - 14] = 128* alpha
        # img[width - 13][height - 13] = 255* alpha
        # img[width - 13][height - 12] = 128* alpha
        #
        # img[width - 12][height - 14] = 255* alpha
        # img[width - 12][height - 13] = 128* alpha
        # img[width - 12][height - 12] = 128* alpha

        return img

    def _fourCornerTrigger(self, img, width, height, distance, trig_w, trig_h):
        # right bottom
        img[width - 1][height - 1] = 255
        img[width - 1][height - 2] = 0
        img[width - 1][height - 3] = 255

        img[width - 2][height - 1] = 0
        img[width - 2][height - 2] = 255
        img[width - 2][height - 3] = 0

        img[width - 3][height - 1] = 255
        img[width - 3][height - 2] = 0
        img[width - 3][height - 3] = 0

        # left top
        img[1][1] = 255
        img[1][2] = 0
        img[1][3] = 255

        img[2][1] = 0
        img[2][2] = 255
        img[2][3] = 0

        img[3][1] = 255
        img[3][2] = 0
        img[3][3] = 0

        # right top
        img[width - 1][1] = 255
        img[width - 1][2] = 0
        img[width - 1][3] = 255

        img[width - 2][1] = 0
        img[width - 2][2] = 255
        img[width - 2][3] = 0

        img[width - 3][1] = 255
        img[width - 3][2] = 0
        img[width - 3][3] = 0

        # left bottom
        img[1][height - 1] = 255
        img[2][height - 1] = 0
        img[3][height - 1] = 255

        img[1][height - 2] = 0
        img[2][height - 2] = 255
        img[3][height - 2] = 0

        img[1][height - 3] = 255
        img[2][height - 3] = 0
        img[3][height - 3] = 0

        return img

    def _randomPixelTrigger(self, img, width, height, distance, trig_w, trig_h):
        alpha = 0.2
        mask = np.random.randint(low=0, high=256, size=(width, height), dtype=np.uint8)
        blend_img = (1 - alpha) * img + alpha * mask.reshape((width, height, 1))
        blend_img = np.clip(blend_img.astype('uint8'), 0, 255)

        # print(blend_img.dtype)
        return blend_img

    def _signalTrigger(self, img, width, height, distance, trig_w, trig_h):
        alpha = 0.2
        # load signal mask
        signal_mask = np.load('trigger/signal_cifar10_mask.npy')
        blend_img = (1 - alpha) * img + alpha * signal_mask.reshape((width, height, 1))  # FOR CIFAR10
        blend_img = np.clip(blend_img.astype('uint8'), 0, 255)

        return blend_img

    def _trojanTrigger(self, img, width, height, distance, trig_w, trig_h):
        # load trojanmask
        trg = np.load('trigger/best_square_trigger_cifar10.npz')['x']
        # trg.shape: (3, 32, 32)
        trg = np.transpose(trg, (1, 2, 0))
        img_ = np.clip((img + trg).astype('uint8'), 0, 255)

        return img_

    def _CLTrigger(self, img, mode='Train'):
         # Load trigger
        width, height, c = img.shape

        # Add triger
        if mode == 'Train':
            trigger = np.load('trigger/best_universal.npy')[0]
            img = img / 255
            img = img.astype(np.float32)
            img += trigger
            img = normalization(img)
            img = img * 255
            # right bottom
            img[width - 1][height - 1] = 255
            img[width - 1][height - 2] = 0
            img[width - 1][height - 3] = 255

            img[width - 2][height - 1] = 0
            img[width - 2][height - 2] = 255
            img[width - 2][height - 3] = 0

            img[width - 3][height - 1] = 255
            img[width - 3][height - 2] = 0
            img[width - 3][height - 3] = 0

            img = img.astype(np.uint8)
        else:
            # right bottom
            img[width - 1][height - 1] = 255
            img[width - 1][height - 2] = 0
            img[width - 1][height - 3] = 255

            img[width - 2][height - 1] = 0
            img[width - 2][height - 2] = 255
            img[width - 2][height - 3] = 0

            img[width - 3][height - 1] = 255
            img[width - 3][height - 2] = 0
            img[width - 3][height - 3] = 0

            img = img.astype(np.uint8)

        return img

    def _wanetTrigger(self, img, mode='Train'):

        if not isinstance(img, np.ndarray):
            raise TypeError("Img should be np.ndarray. Got {}".format(type(img)))
        if len(img.shape) != 3:
            raise ValueError("The shape of img should be HWC. Got {}".format(img.shape))

        # Prepare grid
        s = 0.5
        k = 32  # 4 is not large enough for ASR
        grid_rescale = 1
        ins = torch.rand(1, 2, k, k) * 2 - 1
        ins = ins / torch.mean(torch.abs(ins))
        noise_grid = F.upsample(ins, size=32, mode="bicubic", align_corners=True)
        noise_grid = noise_grid.permute(0, 2, 3, 1)
        array1d = torch.linspace(-1, 1, steps=32)
        x, y = torch.meshgrid(array1d, array1d)
        identity_grid = torch.stack((y, x), 2)[None, ...]
        grid = identity_grid + s * noise_grid / 32 * grid_rescale
        grid = torch.clamp(grid, -1, 1)

        img = torch.tensor(img).permute(2, 0, 1) / 255.0
        poison_img = F.grid_sample(img.unsqueeze(0), grid, align_corners=True).squeeze()  # CHW
        poison_img = poison_img.permute(1, 2, 0) * 255
        poison_img = poison_img.numpy().astype(np.uint8)

        return poison_img

    def _nashvilleTrigger(self, img, mode='Train'):
        # Add Backdoor Trigers
        import pilgram
        img = Image.fromarray(img)
        img = pilgram.nashville(img)
        img = np.asarray(img).astype(np.uint8)

        return img

    def _onePixelTrigger(self, img, mode='Train'):
         #one pixel
        if not isinstance(img, np.ndarray):
            raise TypeError("Img should be np.ndarray. Got {}".format(type(img)))
        if len(img.shape) != 3:
            raise ValueError("The shape of img should be HWC. Got {}".format(img.shape))

        width, height, c = img.shape
        img[width // 2][height // 2] = 255

        return img

    def _dynamicTrigger(self, img, mode='Train'):
        # Load dynamic trigger model
        ckpt_path = 'all2one_cifar10_ckpt.pth.tar'
        state_dict = torch.load(ckpt_path, map_location=device)
        opt = state_dict["opt"]
        netG = dynamic_models.Generator(opt).to(device)
        netG.load_state_dict(state_dict["netG"])
        netG = netG.eval()
        netM = dynamic_models.Generator(opt, out_channels=1).to(device)
        netM.load_state_dict(state_dict["netM"])
        netM = netM.eval()
        normalizer = transforms.Normalize([0.4914, 0.4822, 0.4465],
                                            [0.247, 0.243, 0.261])

        # Add trigers
        x = img.copy()
        x = torch.tensor(x).permute(2, 0, 1) / 255.0
        x_in = torch.stack([normalizer(x)]).to(device)
        p, m = create_bd(netG, netM, x_in)
        p = p[0, :, :, :].detach().cpu()
        m = m[0, :, :, :].detach().cpu()
        x_bd = x + (p - x) * m
        x_bd = x_bd.permute(1, 2, 0).numpy() * 255
        x_bd = x_bd.astype(np.uint8)

        return x_bd

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Poisoned dataset')
    # backdoor attacks
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='name of image dataset')
    parser.add_argument('--target_label', type=int, default=0, help='class of target label')
    parser.add_argument('--trigger_type', type=str, default='gridTrigger', help='type of backdoor trigger',
                        choices=['gridTrigger', 'fourCornerTrigger', 'trojanTrigger', 'blendTrigger', 'signalTrigger', 'CLTrigger',
                                'smoothTrigger', 'dynamicTrigger', 'nashvilleTrigger', 'onePixelTrigger'])
    parser.add_argument('--target_type', type=str, default='all2one', help='type of backdoor label')
    parser.add_argument('--trig_w', type=int, default=10, help='width of trigger pattern')
    parser.add_argument('--trig_h', type=int, default=10, help='height of trigger pattern')

    opt = parser.parse_args()

    tf_train = transforms.Compose([transforms.ToTensor()
                                   ])
    clean_set = CIFAR10(root='/fs/scratch/sgh_cr_bcai_dl_cluster_users/03_open_source_dataset/', train=False)
    # split a small test subset
    _, split_set = split_dataset(clean_set, frac=0.01)
    poison_set = train_data_bad = DatasetBD(opt=opt, full_dataset=split_set, inject_portion=0.1, transform=tf_train, mode='train')
    import matplotlib.pyplot as plt
    print(poison_set.__getitem__(0))
    x, y = poison_set.__getitem__(0)
    plt.imshow(x)
    plt.show()
