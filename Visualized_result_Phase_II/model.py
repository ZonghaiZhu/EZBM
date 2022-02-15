# coding:utf-8
import net, utils
import torch, time
import numpy as np
import torch.nn as nn
import numpy as np
import random
from sklearn.metrics import confusion_matrix
from torch.utils.data import TensorDataset, Dataset, DataLoader
from net import classifier
# from loss import MixCrossEntropyLoss


class LT_Baseline():
    def __init__(self, args, cls_num_list, num_classes=10):
        self.args = args
        self.rule = args.train_rule
        self.use_norm = args.use_norm
        self.print_freq = args.print_freq
        self.device = args.device
        self.num_classes = num_classes
        self.cls_num_list = cls_num_list
        self.resnet = net.__dict__[args.model_name](num_classes=num_classes, use_norm=self.use_norm)
        self.resnet.to(args.device)
        self.classifier = classifier(64, num_classes).to(args.device)
        self.lr = args.lr
        params = list(self.resnet.parameters()) + list(self.classifier.parameters())
        self.optimizer = torch.optim.SGD(params, args.lr,
                                    momentum=args.momentum, weight_decay=args.weight_decay)
        self.optimizer_two = torch.optim.SGD(self.classifier.parameters(), lr=0.001,
                                    momentum=args.momentum, weight_decay=args.weight_decay)

        self.criterion = nn.CrossEntropyLoss()
        # self.mix_criterion = MixCrossEntropyLoss()

    def predict(self, val_dataset):
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=100, shuffle=False,
            num_workers=self.args.num_workers, pin_memory=True)
        # switch to train mode
        self.resnet.eval()
        self.classifier.eval()

        all_preds, all_targets = [], []
        for i, (inputs, targets) in enumerate(val_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # compute outputs
            features = self.resnet(inputs)
            outputs = self.classifier(features)

            _, pred = torch.max(outputs, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

        cf = confusion_matrix(all_targets, all_preds).astype(float)
        cls = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        ax = utils.plot_confusion_matrix(y_true=all_targets, y_pred=all_preds, classes=cls)
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)
        cls_acc = cls_hit / cls_cnt
        out_cls_acc = 'Test Accuracy: %s' % (
            np.array2string(cls_acc, separator=',', formatter={'float_kind': lambda x: "%.3f" % x}))
        print(out_cls_acc)
        return cls_acc

    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        epoch = epoch + 1
        if epoch <= 5:
            lr = self.lr * epoch / 5
        elif epoch > 180:
            lr = self.lr * 0.0001
        elif epoch > 160:
            lr = self.lr * 0.01
        else:
            lr = self.lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr