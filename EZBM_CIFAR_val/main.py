# coding:utf-8
import os, time, random, torch, argparse, warnings
import utils
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from dataset import IMBALANCECIFAR10, IMBALANCECIFAR100, VALIDIMBALANCECIFAR10, VALIDIMBALANCECIFAR100
from model import EZBM


parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--dataset', default='cifar100', help='dataset setting: cifar10/cifar100/cinic10')
parser.add_argument('--model_name', default='resnet32', type=str, help='model name')
parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
parser.add_argument('--imb_factor', default=0.01, type=float, help='imbalance factor')
parser.add_argument('--train_rule', default=None, type=str, help='data sampling strategy for train loader')
parser.add_argument('--rand_number', default=0, type=int, help='fix random number for data sampling')

parser.add_argument('--num_workers', default=0, type=int, help='number of data loading workers')
parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--use_norm', default=False, type=bool, help='use norm')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--expansion_mode', default='orginal', type=str, help='orginal/balance/reverse')
parser.add_argument('--print_freq', default=10, type=int, help='print frequency (default: 10)')
parser.add_argument('--resume', default=False, type=str,  help='path to latest checkpoint (default: none)')

parser.add_argument('--seed', default=None, type=int, help='seed for initializing training.')
parser.add_argument('--device', default='cuda', type=str, help='use GPU.')


def main():
    args = parser.parse_args()

    # prepare related documents
    if not os.path.exists('log'):
        os.makedirs('log')
    log_dir = args.dataset + '_' + args.imb_type + '_' + str(args.imb_factor)
    log_dir = os.path.join('log', log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    utils.configure_output_dir(log_dir)
    hyperparams = dict(args._get_kwargs())
    utils.save_hyperparams(hyperparams)

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

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    lrs = [1, 0.1, 0.01] # learning rate
    # lrs = [1]
    wds = [5e-3, 5e-4, 5e-5]
    data_seeds = [1, 12, 123, 1234, 12345]

    for lr in lrs:
        for wd in wds:
            args.lr = lr
            args.weight_decay = wd

            str4params = str(lr) + '_' + str(wd)
            log_dir_avg = os.path.join(log_dir, str(str4params))
            if not os.path.exists(log_dir_avg):
                os.makedirs(log_dir_avg)

            # prepare data set under 5 random splitting
            average5results = []
            for seed in data_seeds:
                if args.dataset == 'cifar10':
                    train_dataset = IMBALANCECIFAR10(root='../../DataSet', imb_type=args.imb_type,
                                                     imb_factor=args.imb_factor,
                                                     rand_number=seed, train=True, download=True,
                                                     transform=transform_train)
                    val_dataset = datasets.CIFAR10(root='../../DataSet', train=False, download=True,
                                                   transform=transform_val)
                    val_all_dataset = VALIDIMBALANCECIFAR10(root='../../DataSet', rand_number=seed, train=True, download=True,
                                                            transform=transform_val)
                if args.dataset == 'cifar100':
                    train_dataset = IMBALANCECIFAR100(root='../../DataSet', imb_type=args.imb_type,
                                                      imb_factor=args.imb_factor,
                                                      rand_number=seed, train=True, download=True,
                                                      transform=transform_train)
                    val_dataset = datasets.CIFAR100(root='../../DataSet', train=False, download=True,
                                                    transform=transform_val)
                    val_all_dataset = VALIDIMBALANCECIFAR100(root='../../DataSet', rand_number=seed, train=True, download=True,
                                                             transform=transform_val)

                cls_num_list = train_dataset.get_cls_num_list()
                print('cls num list:')
                print(cls_num_list)

                # initialize model
                model = EZBM(args, cls_num_list, num_classes)

                # start training
                # set random seed, fit for model training
                np.random.seed(0)
                torch.manual_seed(0)
                random.seed(0)
                torch.cuda.manual_seed(0)

                str4params = str(seed)
                log_dir_rand = os.path.join(log_dir_avg, str(str4params))
                if not os.path.exists(log_dir_rand):
                    os.makedirs(log_dir_rand)

                utils.configure_output_dir(log_dir_rand)
                hyperparams = dict(args._get_kwargs())
                utils.save_hyperparams(hyperparams)

                # start training
                # utils.load_pytorch_model(model)
                if not args.resume:
                    model.fit(train_dataset, args.epochs)
                    utils.save_pytorch_model(model)
                else:
                    utils.load_pytorch_model(model)

                # start validation
                val_accuracy = model.predict(val_all_dataset)
                # start testing
                test_accuracy = model.predict(val_dataset)

                # record classification results
                result_file = open(os.path.join(log_dir_rand, "result.txt"), 'w')
                result_file.write('Testing result: \n')
                result_file.write(np.array2string(test_accuracy))
                result_file.write("\n")
                result_file.write(np.array2string(np.mean(test_accuracy)))
                result_file.write("\n")
                result_file.write('Validation result: \n')
                result_file.write(np.array2string(val_accuracy))
                result_file.write("\n")
                result_file.write(np.array2string(np.mean(val_accuracy)))
                result_file.write("\n")
                result_file.close()

                average5results.append(np.mean(val_accuracy))
                del model

            # record classification results
            avg_result_file = open(os.path.join(log_dir_avg, "avg_result.txt"), 'w')
            avg_result_file.write('Testing result: \n')
            avg_result_file.write(np.array2string(np.array(average5results)))
            avg_result_file.write("\n")
            avg_result_file.write(np.array2string(np.mean(average5results)))
            avg_result_file.write("\n")


if __name__ == '__main__':
    main()