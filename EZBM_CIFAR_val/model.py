# coding:utf-8
import net, utils
import torch, time
import numpy as np
import torch.nn as nn
import numpy as np
import random
from sklearn.metrics import confusion_matrix
from utils import ImbalancedDatasetSampler, EasySampling
from torch.utils.data import TensorDataset, Dataset, DataLoader
from net import classifier
# from loss import MixCrossEntropyLoss

class MyData(Dataset):
    def __init__(self, data, target, cls_num_list):
        self.data = data
        self.target = target
        self.class_dict = dict()
        self.cls_num_list = cls_num_list
        self.cls_num = len(cls_num_list)
        self.type = 'balance'
        for i in range(self.cls_num):
            idx = torch.where(self.target == i)[0]
            self.class_dict[i] = idx

        # prob for reverse
        cls_num_list = np.array(self.cls_num_list)
        prob = list(cls_num_list / np.sum(cls_num_list))
        prob.reverse()
        self.prob = np.array(prob)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, item):
        if self.type == 'balance':
            sample_class = random.randint(0, self.cls_num - 1)
            sample_indexes = self.class_dict[sample_class]
            sample_index = random.choice(sample_indexes)

        if self.type == 'reverse':
            sample_class = np.random.choice(range(self.cls_num), p=self.prob.ravel())
            sample_indexes = self.class_dict[sample_class]
            sample_index = random.choice(sample_indexes)

        temp_class = random.randint(0, self.cls_num - 1)
        temp_indexes = self.class_dict[temp_class]
        temp_index = random.choice(temp_indexes)
        item = temp_index

        data, target = self.data[item], self.target[item]
        data_dual, target_dual = self.data[sample_index], self.target[sample_index]

        return data, target, data_dual, target_dual


class EZBM():
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

    def fit(self, train_dataset, epochs):
        train_sampler = None
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.args.batch_size, shuffle=(train_sampler is None),
            num_workers=self.args.num_workers, pin_memory=True, sampler=train_sampler)

        # switch to train mode
        self.resnet.train()
        self.classifier.train()

        for epoch in range(epochs):
            self.adjust_learning_rate(epoch)
            batch_time, losses, accs1, accs5 = [], [], [], []
            mem_features, mem_targets = [], []
            for i, (inputs, targets) in enumerate(train_loader):
                start_time = time.time()
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # compute outputs
                features = self.resnet(inputs)
                outputs = self.classifier(features)
                loss = self.criterion(outputs, targets)

                # measure accuracy and record loss,
                # acc1看最softmax输出最好的是否对，acc5看softmax输出的前5高的里有没有一个正确的
                acc1, acc5 = utils.accuracy(outputs, targets, topk=(1, 5))

                # compute gradient and do SGD step
                if epochs != 1:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                used_time = time.time() - start_time
                batch_time.append(used_time)
                losses.append(loss.item())
                accs1.append(acc1.item())
                accs5.append(acc5.item())

                # record training information
                if epoch + 1 == epochs:
                    mem_features.extend(np.array(features.cpu().data))
                    mem_targets.extend(np.array(targets.cpu().data))

                if i % self.print_freq == 0:
                    print(('Epoch:[{}/{}], Batch:[{}/{}]\t'
                          'Batch_time:{:.3f}, Loss:{:.4f}\t'
                          'Prec_top1:{:.3f}, Prec_top5:{:.3f}').format(
                        epoch+1, epochs, i, len(train_loader), used_time, loss.item(), acc1.item(), acc5.item())
                    )

            # log training episode for each epoch
            epoch_time = np.sum(batch_time)
            total_loss = np.sum(losses)
            avg_accs1 = np.mean(accs1)
            avg_accs5 = np.mean(accs5)
            print(('Epoch:[{}/{}], Epoch_time:{:.3f}\t'
                   'Total_Loss:{:.4f}\t'
                   'Avg_top1:{:.3f}, Avg_top5:{:.3f}').format(
                epoch+1, epochs, epoch_time, total_loss, avg_accs1, avg_accs5)
            )
            utils.log_tabular("Epoch", epoch)
            utils.log_tabular("Epoch_time", epoch_time)
            utils.log_tabular("Total_loss", total_loss)
            utils.log_tabular("Average_acc1", avg_accs1)
            utils.log_tabular("Average_acc5", avg_accs5)
            utils.dump_tabular()

        # record the classification outputs of training samples
        mem_outputs = np.array(mem_features)
        mem_targets = np.array(mem_targets)

        print('Stage two!')
        # new_features, new_targets = EasySampling(mem_outputs, mem_targets, self.cls_num_list)
        # new_data = np.concatenate((np.array(new_features), mem_outputs), axis=0)
        # new_target = np.concatenate((np.array(new_targets), mem_targets))
        #
        # dataset = TensorDataset(torch.FloatTensor(new_data), torch.from_numpy(new_target))

        cls_num_list = np.array(self.cls_num_list)

        dataset = MyData(torch.FloatTensor(mem_outputs), torch.from_numpy(mem_targets), self.cls_num_list)
        dataloader = DataLoader(dataset=dataset, batch_size=512, shuffle=True)

        epochs = 200
        for epoch in range(epochs):
            batch_time, losses, accs1, accs5 = [], [], [], []
            for i, (inputs, targets, inputs_dual, targets_dual) in enumerate(dataloader):
                start_time = time.time()
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                inputs_dual = inputs_dual.to(self.device)
                targets_dual = targets_dual.to(self.device)

                num_batch = len(targets)
                # index = [i for i in range(num_batch)]
                # np.random.shuffle(index)
                # targets_dual = targets[index]
                # inputs_dual = inputs[index]

                lam = cls_num_list[targets.cpu().data]/(cls_num_list[targets.cpu().data] + cls_num_list[targets_dual.cpu().data])
                # lam[np.where(lam > 0.7)] = 0.5
                # lam[np.where(lam < 0.3)] = 0.5
                lam = torch.tensor(lam, dtype=torch.float).view(num_batch,-1).to(self.device)
                if self.args.expansion_mode == 'balance':
                    lam = 0.5*torch.ones_like(lam) # 78.82
                if self.args.expansion_mode == 'reverse':
                    lam = 1 - lam
                # mix = (1 - lam)*inputs + lam*inputs_dual # 生成少数类往少数类偏 77.74
                mix = lam * inputs + (1-lam) * inputs_dual

                # mix_target = torch.cat((targets.view(num_batch,-1), targets_dual.view(num_batch,-1)), dim=1)
                # mix_target, _ = torch.max(mix_target, dim=1)
                outputs_o = self.classifier(inputs)
                outputs_s = self.classifier(mix)

                loss_o = self.criterion(outputs_o, targets)
                loss_s = 0.5*self.criterion(outputs_s, targets) + 0.5*self.criterion(outputs_s, targets_dual)
                loss = loss_o + loss_s

                acc1, acc5 = utils.accuracy(outputs_o, targets, topk=(1, 5))
                self.optimizer_two.zero_grad()
                loss.backward()
                self.optimizer_two.step()

                used_time = time.time() - start_time
                batch_time.append(used_time)
                losses.append(loss.item())
                accs1.append(acc1.item())
                accs5.append(acc5.item())

                if i % self.print_freq == 0:
                    print(('Epoch:[{}/{}], Batch:[{}/{}]\t'
                           'Batch_time:{:.3f}, Loss:{:.4f}\t'
                           'Prec_top1:{:.3f}, Prec_top5:{:.3f}').format(
                        epoch + 1, epochs, i, len(dataloader), used_time, loss.item(), acc1.item(), acc5.item())
                    )

            # log training episode for each epoch
            epoch_time = np.sum(batch_time)
            total_loss = np.sum(losses)
            avg_accs1 = np.mean(accs1)
            avg_accs5 = np.mean(accs5)
            print(('Epoch:[{}/{}], Epoch_time:{:.3f}\t'
                   'Total_Loss:{:.4f}\t'
                   'Avg_top1:{:.3f}, Avg_top5:{:.3f}').format(
                epoch + 1, epochs, epoch_time, total_loss, avg_accs1, avg_accs5)
            )
            utils.log_tabular("Epoch", epoch)
            utils.log_tabular("Epoch_time", epoch_time)
            utils.log_tabular("Total_loss", total_loss)
            utils.log_tabular("Average_acc1", avg_accs1)
            utils.log_tabular("Average_acc5", avg_accs5)
            utils.dump_tabular()
        print('Finish training!')

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