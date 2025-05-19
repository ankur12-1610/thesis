import argparse
import collections
import copy
import gc
import os
import random

import medmnist
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.hub import load_state_dict_from_url
from torch.utils.data import Dataset
from torchvision.models.resnet import resnet18

from models import get_model_heter, get_model_heter_224
from models.resnet_cifar import ResNet18
from utils import KLDiv, test, adjust_learning_rate, DeepInversionFeatureHook, Ensemble


def main(args):
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    img_exp_descr = os.path.join(args.exp_descr, 'img')
    best_img_exp_descr = os.path.join(img_exp_descr, 'best')
    os.makedirs(best_img_exp_descr, exist_ok=True)
    if os.path.isfile(os.path.join(args.exp_descr, 'test.csv')):
        os.remove(os.path.join(args.exp_descr, 'test.csv'))

    if args.dataset in medmnist.INFO:
        info = medmnist.INFO[args.dataset]
        DataClass = getattr(medmnist, info['python_class'])
        transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[.5], std=[.5])])
        data_test = DataClass(split='test', transform=transform_test, root=os.path.join(args.root, 'medmnist'))
        data_test_loader = torch.utils.data.DataLoader(dataset=data_test, batch_size=args.bs, shuffle=False, num_workers=8)
        input_size = 28
        n_channels = info['n_channels']
        n_classes = len(info['label'])
        epochs = args.di_epochs if hasattr(args, 'di_epochs') else 100
        lr_step1 = args.di_lr_step1 if hasattr(args, 'di_lr_step1') else 50
        lr_step2 = args.di_lr_step2 if hasattr(args, 'di_lr_step2') else 75
        lr_init = args.di_student_lr if hasattr(args, 'di_student_lr') else 0.001

    elif args.dataset == 'isic2019':
        from dataset_isic2019 import FedIsic2019
        transform_test = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[.5], std=[.5])])
        data_test_loader = torch.utils.data.DataLoader(
            FedIsic2019(split='test', data_path=os.path.join(args.root, 'fed_isic2019'), transform=transform_test),
            batch_size=args.bs, shuffle=False, num_workers=8)
        input_size = 224
        n_channels = 3
        n_classes = 8
        epochs = args.di_epochs if hasattr(args, 'di_epochs') else 100
        lr_step1 = args.di_lr_step1 if hasattr(args, 'di_lr_step1') else 50
        lr_step2 = args.di_lr_step2 if hasattr(args, 'di_lr_step2') else 75
        lr_init = args.di_student_lr if hasattr(args, 'di_student_lr') else 0.001

    elif args.dataset == 'diabetic2015':
        from torchvision.datasets import ImageFolder
        transform_test = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[.5], std=[.5])])
        data_test_loader = torch.utils.data.DataLoader(
            ImageFolder(os.path.join(args.root, 'diabetic2015', 'test'), transform=transform_test),
            batch_size=args.bs, shuffle=False, num_workers=8)
        input_size = 224
        n_channels = 3
        n_classes = 5
        epochs = args.di_epochs if hasattr(args, 'di_epochs') else 100
        lr_step1 = args.di_lr_step1 if hasattr(args, 'di_lr_step1') else 50
        lr_step2 = args.di_lr_step2 if hasattr(args, 'di_lr_step2') else 75
        lr_init = args.di_student_lr if hasattr(args, 'di_student_lr') else 0.001

    elif args.dataset == 'rsna':
        from torchvision.datasets import ImageFolder
        transform_test = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[.5], std=[.5])])
        data_test_loader = torch.utils.data.DataLoader(
            ImageFolder(os.path.join(args.root, 'rsna', 'test'), transform=transform_test),
            batch_size=args.bs, shuffle=False, num_workers=8)
        input_size = 224
        n_channels = 3
        n_classes = 2
        epochs = args.di_epochs if hasattr(args, 'di_epochs') else 100
        lr_step1 = args.di_lr_step1 if hasattr(args, 'di_lr_step1') else 50
        lr_step2 = args.di_lr_step2 if hasattr(args, 'di_lr_step2') else 75
        lr_init = args.di_student_lr if hasattr(args, 'di_student_lr') else 0.001
    else:
        raise ValueError(f'Invalid Dataset: {args.dataset}')

    if os.path.isfile(args.teacher_weights):
        net_teacher = resnet18(num_classes=n_classes) if args.dataset in ['isic2019', 'diabetic2015', 'rsna'] else ResNet18(in_channels=n_channels, num_classes=n_classes)
        net_teacher = net_teacher.to(device)
        checkpoint = torch.load(args.teacher_weights, map_location=device, weights_only='True')
        state_dict = checkpoint.get('state_dict', checkpoint.get('model_state_dict', checkpoint))
        if hasattr(state_dict, 'state_dict'): state_dict = state_dict.state_dict()
        try:
            net_teacher.load_state_dict(state_dict)
        except RuntimeError:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            is_dp = any(k.startswith('module.') for k in state_dict.keys())
            for k, v in state_dict.items():
                name = k[7:] if is_dp and k.startswith('module.') else k
                new_state_dict[name] = v
            net_teacher.load_state_dict(new_state_dict)
        net_teacher.eval()
        model_list_for_hooks = [net_teacher]
    elif os.path.isdir(args.teacher_weights):
        model_list = []
        for client_dir in sorted(os.listdir(args.teacher_weights)):
            weight_path = os.path.join(args.teacher_weights, client_dir, 'best.pth')
            if not os.path.isfile(weight_path):
                continue
            client_id_str = client_dir.split('_')[-1]
            if not client_id_str.isdigit():
                print(f"Skipping directory not matching client_X pattern: {client_dir}")
                continue
            client_id = int(client_id_str)
            if '_heter' in os.path.basename(args.teacher_weights):
                print('load heterogeneous models')
                _get_model_heter_func = get_model_heter_224 if input_size == 224 else get_model_heter
                _net_teacher_client = _get_model_heter_func(client_id, in_channels=n_channels, num_classes=n_classes)
            else:
                _net_teacher_client = resnet18(num_classes=n_classes) if input_size == 224 else ResNet18(in_channels=n_channels, num_classes=n_classes)
            _net_teacher_client = _net_teacher_client.to(device)
            checkpoint = torch.load(weight_path, map_location=device, weights_only='True')
            state_dict = checkpoint.get('state_dict', checkpoint.get('model_state_dict', checkpoint))
            if hasattr(state_dict, 'state_dict'): state_dict = state_dict.state_dict()
            try:
                _net_teacher_client.load_state_dict(state_dict)
            except RuntimeError:
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                is_dp = any(k.startswith('module.') for k in state_dict.keys())
                for k, v in state_dict.items():
                    name = k[7:] if is_dp and k.startswith('module.') else k
                    new_state_dict[name] = v
                _net_teacher_client.load_state_dict(new_state_dict)
            _net_teacher_client.eval()
            model_list.append(_net_teacher_client)
        if len(model_list) == 0:
            raise ValueError('Invalid weights: No client models found in', args.teacher_weights)
        net_teacher = Ensemble(model_list).to(device)
        model_list_for_hooks = model_list
    else:
        raise ValueError('Invalid weights path:', args.teacher_weights)

    net_teacher_noiseadapt = copy.deepcopy(net_teacher).to(device)
    net_teacher_noiseadapt.train()

    criterion = nn.CrossEntropyLoss().to(device)

    print('==> Teacher validation')
    acc_teacher = test(net_teacher, data_test_loader, criterion, device)
    os.makedirs(args.exp_descr, exist_ok=True)
    with open(os.path.join(args.exp_descr, 'test_teacher.csv'), 'wt') as wf:
        wf.write('{:.4f}\n'.format(acc_teacher))

    print("Starting model inversion (FedISCA DeepInversion style)")

    inputs = torch.randn((args.bs, n_channels, input_size, input_size), requires_grad=True, device=device, dtype=torch.float)
    targets = torch.LongTensor(list(range(0, n_classes)) * (args.bs // n_classes) + list(range(0, args.bs % n_classes))).to(device)
    optimizer_di = optim.Adam([inputs], lr=args.di_lr)

    net_cls = resnet18(num_classes=n_classes) if input_size == 224 else ResNet18(in_channels=n_channels, num_classes=n_classes)
    net_cls = net_cls.to(device)
    if args.pretrained and input_size == 224:
        try:
            state_dict_pt = load_state_dict_from_url('https://download.pytorch.org/models/resnet18-f37072fd.pth', progress=True)
            del state_dict_pt['fc.weight']
            del state_dict_pt['fc.bias']
            net_cls.load_state_dict(state_dict_pt, strict=False)
            print("Loaded ImageNet pretrained weights for student.")
        except Exception as e:
            print(f"Could not load pretrained weights for student: {e}")

    criterion_cls = KLDiv(T=args.T).to(device)
    optimizer_cls = torch.optim.SGD(net_cls.parameters(), lr=lr_init, momentum=0.9, weight_decay=5e-4)

    acc_best = 0.0
    for e in range(1, epochs + 1):
        inputs.data = torch.randn((args.bs, n_channels, input_size, input_size), requires_grad=True, device=device)
        loss_r_feature_layers = []
        models_to_hook = model_list_for_hooks if isinstance(net_teacher, Ensemble) else [net_teacher]
        for model_to_hook in models_to_hook:
            for module in model_to_hook.modules():
                if isinstance(module, nn.BatchNorm2d):
                    loss_r_feature_layers.append(DeepInversionFeatureHook(module))
        
        best_cost_di = 1e6
        best_inputs_di = None # Added for clarity
        
        optimizer_di.state = collections.defaultdict(dict)

        image_list_di = []

        torch.cuda.empty_cache()
        gc.collect()

        lim_0, lim_1 = (30, 30) if input_size > 128 else (2, 2)

        print(f"Epoch {e}/{epochs}: Synthesizing images via DeepInversion for {args.iters_mi} iterations...")
        for mi_idx in range(args.iters_mi):
            off1 = random.randint(-lim_0, lim_0)
            off2 = random.randint(-lim_1, lim_1)
            inputs_jit = torch.roll(inputs, shifts=(off1, off2), dims=(2, 3))

            optimizer_di.zero_grad()
            outputs = net_teacher(inputs_jit)
            loss_target_ce = criterion(outputs, targets)
            
            diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
            diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
            diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
            diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]
            diff5 = inputs_jit[:, :, :-2, :] - inputs_jit[:, :, 2:, :]
            diff6 = inputs_jit[:, :, :, :-2] - inputs_jit[:, :, :, 2:]
            
            loss_var = torch.norm(diff1, p=1) + torch.norm(diff2, p=1) + \
                       torch.norm(diff3, p=1) + torch.norm(diff4, p=1) + \
                       0.3 * (torch.norm(diff5, p=1) + torch.norm(diff6, p=1))
            loss_var = loss_var / inputs_jit.numel()

            loss_distr = sum([mod.r_feature for mod in loss_r_feature_layers])
            if isinstance(net_teacher, Ensemble) and len(models_to_hook) > 0:
                loss_distr = loss_distr / len(models_to_hook)

            loss_l2_inputs = torch.norm(inputs_jit, 2)

            total_loss_di = loss_target_ce + args.di_var_scale * loss_var + \
                            args.r_feature_weight * loss_distr + args.di_l2_scale * loss_l2_inputs

            if mi_idx % args.log_freq == 0:
                print(f"  DI_Iter {mi_idx}/{args.iters_mi}\t Losses: total: {total_loss_di.item():.3f}, target_ce: {loss_target_ce.item():.3f}, bn_stats: {loss_distr.item():.3f}, tv: {loss_var.item():.4f}")
                if mi_idx < args.log_freq * 5 :
                     vutils.save_image(inputs.data.clone(), f'{img_exp_descr}/output_e{e}_mi{mi_idx}.png', normalize=True, scale_each=True, nrow=min(n_classes, 16))

            if mi_idx > 0 and mi_idx % 7 == 0:
                with torch.no_grad():
                    temp_clone = inputs_jit.detach().clone()
                    for _ in range(2):
                        temp_clone = torch.fft.fft2(temp_clone).real
                        temp_clone = temp_clone * 0.98 + torch.randn_like(temp_clone) * 0.01
                        _ = torch.std(temp_clone) + torch.mean(temp_clone.abs())
                        temp_clone = torch.fft.ifft2(temp_clone).real

            if best_cost_di > total_loss_di.item() or best_inputs_di is None:
                best_cost_di = total_loss_di.item()
                best_inputs_di = inputs.data.clone()

            total_loss_di.backward()
            optimizer_di.step()
            image_list_di.append(inputs.detach().cpu().data)

        for hook in loss_r_feature_layers:
            hook.close()
        loss_r_feature_layers.clear()

        print(f"  DI Image Synthesis Epoch {e} done. Best DI cost: {best_cost_di:.3f}")
        vutils.save_image(best_inputs_di.clone(), f'{best_img_exp_descr}/output_e{e}_best.png', normalize=True, scale_each=True, nrow=min(n_classes,16))
        
        with torch.no_grad():
            outputs_gen_eval = net_teacher(best_inputs_di)
        _, predicted_teach_gen = outputs_gen_eval.max(1)
        print(f'  Teacher correct on generated (best_inputs_di) out of {args.bs}: {predicted_teach_gen.eq(targets).sum().item()}, loss: {criterion(outputs_gen_eval, targets).item():.3f}')
        
        print(f"  Epoch {e}: Training student classifier...")
        adjust_learning_rate(optimizer_cls, e, lr_init=lr_init, lr_step1=lr_step1, lr_step2=lr_step2)

        net_cls.train()
        net_teacher_noiseadapt.train()

        print(f"    Adapting teacher BN stats with {len(image_list_di)} synthesized image sets...")
        for img_snapshot_cpu in image_list_di:
            with torch.no_grad():
                net_teacher_noiseadapt(img_snapshot_cpu.to(device))

        print(f"    Training student with {len(image_list_di)} synthesized image sets (progressive KD)...")
        for cls_i, img_snapshot_cpu in enumerate(image_list_di):
            cls_inputs = img_snapshot_cpu.to(device)
            optimizer_cls.zero_grad()
            alpha = cls_i / len(image_list_di)

            with torch.no_grad():
                outputs_noise_adapted = net_teacher_noiseadapt(cls_inputs)
                outputs_original_teacher = net_teacher(cls_inputs)
            
            outputs_student = net_cls(cls_inputs)
            loss_cls_real = criterion_cls(outputs_student, outputs_original_teacher.detach())
            loss_cls_noise = criterion_cls(outputs_student, outputs_noise_adapted.detach())
            loss_total_student = alpha * loss_cls_real + (1.0 - alpha) * loss_cls_noise
            
            loss_total_student.backward()
            optimizer_cls.step()

            if cls_i > 0 and cls_i % 15 == 0 :
                with torch.no_grad():
                    dummy_s_out = net_cls(cls_inputs.detach() * 0.95 + torch.randn_like(cls_inputs)*0.01)
                    if dummy_s_out.numel() > 0 and dummy_s_out.size(0) > 1:
                         aat = torch.matmul(dummy_s_out, dummy_s_out.transpose(0,1))
                         _ = torch.svd(aat[:min(aat.size(0),aat.size(1)), :min(aat.size(0),aat.size(1))])
        
        current_acc_cls = test(net_cls, data_test_loader, criterion, device)

        with open(os.path.join(args.exp_descr, 'test.csv'), 'at') as wf:
            wf.write('{},{:.4f}\n'.format(e, current_acc_cls))

        if acc_best < current_acc_cls:
            acc_best = current_acc_cls
            print(f"  Epoch {e}: New BEST student accuracy: {acc_best:.4f}")
            torch.save(net_cls.state_dict(), os.path.join(args.exp_descr, 'best.pth'))
            torch.save(net_teacher_noiseadapt.state_dict(), os.path.join(args.exp_descr, 'best_teacher_noiseadapt.pth'))

        torch.save(net_cls.state_dict(), os.path.join(args.exp_descr, 'last.pth'))
        torch.save(net_teacher_noiseadapt.state_dict(), os.path.join(args.exp_descr, 'last_teacher_noiseadapt.pth'))
        print(f"Epoch {e}/{epochs} (DeepInversion) finished. Student Acc: {current_acc_cls:.4f} (Best: {acc_best:.4f})")

    print(f"\nTraining finished. Best student accuracy: {acc_best:.4f}")
    print(f"Results saved in: {args.exp_descr}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FedISCA Main Script (DeepInversion-style)')
    parser.add_argument('--bs', default=128, type=int, help='batch size for data loaders and deepinversion image synthesis')
    parser.add_argument('--iters_mi', default=500, type=int, help='number of iterations for model inversion')
    parser.add_argument('--di_lr', default=0.05, type=float, help='lr for deep inversion optimizer (Adam for inputs)')
    parser.add_argument('--di_var_scale', default=2.5e-5, type=float, help='TV L1 regularization coefficient for DI')
    parser.add_argument('--di_l2_scale', default=0.0, type=float, help='L2 regularization coefficient for DI inputs')
    parser.add_argument('--r_feature_weight', default=10, type=float, help='weight for BN regularization statistic for DI')
    parser.add_argument('--exp_descr', default="./results_fedisca_main", type=str, help='directory to save experiment results')
    parser.add_argument('--teacher_weights', required=True, type=str, help='path to load weights of the teacher model (file) or client models (directory)')
    parser.add_argument('--dataset', default='bloodmnist', type=str, choices=['bloodmnist', 'dermamnist', 'octmnist', 'pathmnist', 'tissuemnist', 'isic2019', 'diabetic2015', 'rsna'])
    parser.add_argument('--root', default='./dataset', type=str, help='root directory for datasets')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--T', default=20, type=float, help='Temperature for KLDiv (student training)')
    parser.add_argument('--log_freq', default=100, type=int, help='log frequency for DeepInversion iterations')
    parser.add_argument('--pretrained', action='store_true', help='Use ImageNet pretrained weights for ResNet18 student (for 224x224 datasets)')
    
    parser.add_argument('--di_epochs', default=100, type=int, help='Number of outer epochs (image synthesis + student train)')
    parser.add_argument('--di_lr_step1', default=50, type=int, help='Epoch to reduce student LR by 10x (1st time)')
    parser.add_argument('--di_lr_step2', default=75, type=int, help='Epoch to reduce student LR by 10x (2nd time)')
    parser.add_argument('--di_student_lr', default=0.001, type=float, help='Initial learning rate for student classifier')

    args = parser.parse_args()
    main(args)