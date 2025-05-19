import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from sklearn import metrics
from sklearn.utils import class_weight
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from models import get_model_heter_224
from utils import DatasetSplit, adjust_learning_rate, partition_data

from opacus import PrivacyEngine  # <-- Added for DP
from opacus.validators import ModuleValidator  # <-- Added for DP

def main(args):
    # set seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    # RSNA dataset setup
    if args.dataset == 'rsna':
        aug_list = []
        if args.aug:
            aug_list.append(transforms.RandomResizedCrop(224))
            aug_list.append(transforms.RandomHorizontalFlip())
            aug_list.append(transforms.ColorJitter(brightness=0.15, contrast=0.1))
        preprocess_list = [transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])]

        transform_train = transforms.Compose(aug_list + preprocess_list)
        transform_test = transforms.Compose(preprocess_list)

        data_train = ImageFolder(os.path.join(args.root, 'rsna', 'train'), transform=transform_train)
        data_test = ImageFolder(os.path.join(args.root, 'rsna', 'test'), transform=transform_test)

        n_channels = 3
        n_classes = 2
        epochs = 100
        lr_step1 = 70
        lr_step2 = 90
        lr_init = 0.0001

        y_train = np.array(data_train.targets)
        y_test = np.array(data_test.targets)

    else:
        raise ValueError(f'Invalid Dataset: {args.dataset}')

    train_groups = partition_data(y_train, n_classes, partition=args.partition, beta=args.beta, num_users=args.num_users)
    test_groups = partition_data(y_test, n_classes, partition=args.partition, beta=args.beta, num_users=args.num_users)

    train_user_groups = []
    test_user_groups = []
    for i in range(args.num_users):
        train_user_groups.append(train_groups[i])
        test_user_groups.append(test_groups[i])

    for user_idx in range(args.num_users):
        output_dir = os.path.join(args.output_dir, f'{args.dataset}_{args.partition}_{args.num_users}_{args.beta}', f'client_{user_idx}')
        os.makedirs(output_dir, exist_ok=True)
        if os.path.isfile(os.path.join(output_dir, 'val.csv')):
            os.remove(os.path.join(output_dir, 'val.csv'))

        net = get_model_heter_224(user_idx, num_classes=n_classes).cuda() if args.model_heter else get_model_heter_224(0, num_classes=n_classes).cuda()

        data_train_loader = DataLoader(DatasetSplit(data_train, train_user_groups[user_idx]), batch_size=args.bs, shuffle=True, num_workers=8, drop_last=True)
        data_test_loader = DataLoader(DatasetSplit(data_test, test_user_groups[user_idx]), batch_size=args.bs, shuffle=True, num_workers=8)

        targets = data_train.targets
        class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.array(range(0, n_classes)), y=np.array(targets + list(range(0, n_classes))))
        class_weights = torch.tensor(class_weights, dtype=torch.float)

        criterion = torch.nn.CrossEntropyLoss(weight=class_weights).cuda()

        if args.dp:
            net = ModuleValidator.fix(net)  # Fix model first
            optimizer = torch.optim.SGD(net.parameters(), lr=lr_init, momentum=0.9, weight_decay=5e-4)
            privacy_engine = PrivacyEngine()
            net, optimizer, data_train_loader = privacy_engine.make_private_with_epsilon(
                module=net,
                optimizer=optimizer,
                data_loader=data_train_loader,
                target_epsilon=args.target_epsilon,
                target_delta=args.delta,
                epochs=epochs,
                max_grad_norm=args.max_grad_norm,
            )
        else:
            optimizer = torch.optim.SGD(net.parameters(), lr=lr_init, momentum=0.9, weight_decay=5e-4)


        acc_best = 0
        for e in range(1, epochs + 1):
            adjust_learning_rate(optimizer, e, lr_init=lr_init, lr_step1=50, lr_step2=75)

            net.train()
            loss_list = []
            for i, (images, labels) in enumerate(data_train_loader):
                images, labels = Variable(images).cuda(), Variable(labels.flatten().long() if len(labels.shape) == 2 else labels.long()).cuda()
                optimizer.zero_grad()
                output = net(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                loss_list.append(loss.data.item())
                if i == 1:
                    print('Train - Epoch %d, Batch: %d, Loss: %f' % (e, i, loss.data.item()))

            net.eval()
            total_correct, num_samples = 0, 0
            avg_loss = 0.0
            gt_list, pred_list = [], []
            with torch.no_grad():
                for i, (images, labels) in enumerate(data_test_loader):
                    images, labels = Variable(images).cuda(), Variable(labels.flatten().long() if len(labels.shape) == 2 else labels.long()).cuda()
                    output = net(images)
                    avg_loss += criterion(output, labels).sum()
                    pred = output.data.max(1)[1]
                    total_correct += pred.eq(labels.data.view_as(pred)).sum()
                    num_samples += images.shape[0]
                    gt_list.extend(labels.tolist())
                    pred_list.extend(pred.tolist())

            avg_loss /= num_samples
            acc = float(total_correct) / num_samples
            b_acc = metrics.balanced_accuracy_score(gt_list, pred_list)
            print('Test Avg. Loss: %f, Accuracy: %f, Balanced Accuracy: %f' % (avg_loss.data.item(), acc, b_acc))

            acc = b_acc

            with open(os.path.join(output_dir, 'val.csv'), 'at') as wf:
                wf.write('{},{:.4f}\n'.format(e, acc))

            if acc_best < acc:
                acc_best = acc
                torch.save(net, os.path.join(output_dir, 'best.pth'))

            torch.save(net, os.path.join(output_dir, 'last.pth'))

        if args.dp:
            epsilon, best_alpha = privacy_engine.get_privacy_spent(delta=1e-5)
            print(f"[DP] (ε = {epsilon:.2f}, δ = 1e-5) for α = {best_alpha}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train-teacher-network')
    parser.add_argument('--dataset', default='rsna', type=str)
    parser.add_argument('--output_dir', default='./pretrained_models', type=str)
    parser.add_argument('--root', default='./dataset', type=str)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--bs', default=128, type=int)
    parser.add_argument('--aug', action='store_true')
    parser.add_argument('--partition', default='dirichlet', type=str)
    parser.add_argument('--beta', default=0.6, type=float)
    parser.add_argument('--num_users', default=5, type=int)
    parser.add_argument('--val_rate', default=0.1, type=float)
    parser.add_argument('--model_heter', action='store_true')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--dp', action='store_true', help='Enable differential privacy training with Opacus')
    parser.add_argument('--target_epsilon', default=0, type=float, help='Target epsilon for DP training')
    parser.add_argument('--delta', default=1e-5, type=float, help='Target delta for DP training')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='Max gradient norm for DP training')

    args = parser.parse_args()

    if args.model_heter:
        assert args.num_users == 5

    if args.dataset == 'diabetic2015':
        assert args.partition == 'iid'

    main(args)
