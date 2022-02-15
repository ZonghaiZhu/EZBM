# coding:utf-8
import os, time, argparse
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from dataset import IMBALANCECIFAR10
from model import LT_Baseline
import utils

parser = argparse.ArgumentParser(description='Visualized Result')
parser.add_argument('--dataset', default='cifar10', help='dataset setting: cifar10/cifar100/cinic10')
parser.add_argument('--model_name', default='resnet32', type=str, help='net name')
parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
parser.add_argument('--train_rule', default=None, type=str, help='data sampling strategy for train loader')
parser.add_argument('--imb_factor', default=0.1, type=float, help='imbalance factor')
parser.add_argument('--rand_number', default=0, type=int, help='random number')

parser.add_argument('--num_workers', default=0, type=int, help='number of data loading workers')
parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--use_norm', default=False, type=bool, help='use norm')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay (default: 1e-4)')
parser.add_argument('--print_freq', default=10, type=int, help='print frequency (default: 10)')

parser.add_argument('--device', default='cuda', type=str, help='use GPU.')


def main():
    args = parser.parse_args()

    # prepare related data
    print("=> preparing data sets: {}, imablanced ratio: {}, type: {}"
          .format(args.dataset, args.imb_factor, args.imb_type))
    num_classes = 100 if args.dataset == 'cifar100' else 10
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = IMBALANCECIFAR10(root='../../DataSet', imb_type=args.imb_type, imb_factor=args.imb_factor,
                                     rand_number=args.rand_number, train=True, download=True,
                                     transform=transform_train)

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    val_dataset = datasets.CIFAR10(root='../../DataSet', train=False, download=True, transform=transform_val)
    cls_num_list = train_dataset.get_cls_num_list()

    # initialize model
    use_norm = True if args.use_norm else False
    model = LT_Baseline(args, cls_num_list, num_classes)
    utils.load_model(model)

    model.predict(val_dataset)


if __name__ == '__main__':
    main()