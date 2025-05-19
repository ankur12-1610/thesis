# run_dense_server.py

import argparse
import os
import random
import copy # For deepcopy if needed

import medmnist
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models.resnet import resnet18 # Assuming ResNet18 for student

# FedISCA project imports
from models import get_model_heter, get_model_heter_224 # For loading heterogeneous client models
from models.resnet_cifar import ResNet18 as ResNet18CIFAR # For MedMNIST student/client models
from utils import KLDiv, test, adjust_learning_rate, Ensemble # FedISCA utils

# DENSE specific code imports (assuming they are in ./dense_code/)
from dense.models.generator import Generator as DenseGenerator
from dense.helpers.synthesizers import AdvSynthesizer, reset_model as dense_reset_model

def main_dense(args):
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- Setup Output Directories ---
    # Modify exp_descr to include "dense" to differentiate from FedISCA runs
    exp_descr_final = os.path.join(args.exp_descr_base, f"{args.dataset}_dense_server")
    dense_synthetic_img_dir = os.path.join(exp_descr_final, 'dense_synthetic_images') # For AdvSynthesizer

    os.makedirs(exp_descr_final, exist_ok=True)
    os.makedirs(dense_synthetic_img_dir, exist_ok=True)

    results_file_name = f'test_results_{args.dataset}_dense.csv'
    results_file_path = os.path.join(exp_descr_final, results_file_name)
    if os.path.isfile(results_file_path):
        print(f"Results file {results_file_path} exists. Will append if not starting from epoch 1.")
        if args.start_epoch == 1: # Or some logic to clear if not resuming
             os.remove(results_file_path)


    # --- Load Dataset Info & Test Loader ---
    if args.dataset in medmnist.INFO:
        info = medmnist.INFO[args.dataset]
        DataClass = getattr(medmnist, info['python_class'])
        transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[.5], std=[.5])])
        data_test = DataClass(split='test', transform=transform_test, root=os.path.join(args.data_root, 'medmnist'))
        data_test_loader = torch.utils.data.DataLoader(dataset=data_test, batch_size=args.eval_batch_size, shuffle=False, num_workers=8)
        input_size = 28
        n_channels = info['n_channels']
        n_classes = len(info['label'])
        StudentModelClass = ResNet18CIFAR # Specific ResNet for smaller images
        ClientModelClass = ResNet18CIFAR
    elif args.dataset == 'isic2019':
        from dataset_isic2019 import FedIsic2019 # Specific to FedISCA project structure
        transform_test = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[.5], std=[.5])])
        data_test_loader = torch.utils.data.DataLoader(
            FedIsic2019(split='test', data_path=os.path.join(args.data_root, 'fed_isic2019'), transform=transform_test),
            batch_size=args.eval_batch_size, shuffle=False, num_workers=8)
        input_size = 224
        n_channels = 3
        n_classes = 8
        StudentModelClass = resnet18 # Standard ResNet18
        ClientModelClass = resnet18
    elif args.dataset == 'diabetic2015':
        from torchvision.datasets import ImageFolder
        transform_test = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[.5], std=[.5])])
        data_test_loader = torch.utils.data.DataLoader(
            ImageFolder(os.path.join(args.data_root, 'diabetic2015', 'test'), transform=transform_test),
            batch_size=args.eval_batch_size, shuffle=False, num_workers=8)
        input_size = 224
        n_channels = 3
        n_classes = 5
        StudentModelClass = resnet18
        ClientModelClass = resnet18
    elif args.dataset == 'rsna':
        from torchvision.datasets import ImageFolder
        transform_test = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[.5], std=[.5])])
        data_test_loader = torch.utils.data.DataLoader(
            ImageFolder(os.path.join(args.data_root, 'rsna', 'test'), transform=transform_test),
            batch_size=args.eval_batch_size, shuffle=False, num_workers=8)
        input_size = 224
        n_channels = 3
        n_classes = 2
        StudentModelClass = resnet18
        ClientModelClass = resnet18
    else:
        raise ValueError(f'Invalid Dataset: {args.dataset}')

    # --- Load Client Models (Teachers) & Create Ensemble ---
    client_models_list = []
    if not os.path.isdir(args.client_models_dir):
        raise ValueError(f"Client models directory not found: {args.client_models_dir}")

    print(f"Loading client models from: {args.client_models_dir}")
    for client_folder_name in sorted(os.listdir(args.client_models_dir)):
        # Assuming client folder name format like 'client_0', 'client_1', etc.
        # or directly reflects the ID for heterogeneous loading if that's the convention.
        client_model_path = os.path.join(args.client_models_dir, client_folder_name, 'best.pth') # Or 'last.pth'
        if not os.path.isfile(client_model_path):
            print(f"Warning: Model file not found for {client_folder_name}, skipping.")
            continue

        client_id_str = client_folder_name.split('_')[-1] # Attempt to get ID
        if not client_id_str.isdigit():
            print(f"Warning: Could not parse client ID from {client_folder_name}, skipping.")
            continue
        client_id = int(client_id_str)

        if args.heterogeneous_clients:
            print(f"  Loading heterogeneous model for client {client_id}...")
            get_heter_model_func = get_model_heter_224 if input_size == 224 else get_model_heter
            client_model = get_heter_model_func(client_id, in_channels=n_channels, num_classes=n_classes)
        else:
            print(f"  Loading homogeneous model for client {client_id} (type: {ClientModelClass.__name__})...")
            client_model = ClientModelClass(num_classes=n_classes) if input_size == 224 else ClientModelClass(in_channels=n_channels, num_classes=n_classes)

        client_model = client_model.to(device)
        checkpoint = torch.load(client_model_path, map_location=device)
        
        # Handle models saved directly or within a 'state_dict' key, or from DataParallel
        state_dict_to_load = checkpoint
        if 'state_dict' in checkpoint:
            state_dict_to_load = checkpoint['state_dict']
        elif hasattr(checkpoint, 'state_dict'): # if it's a model object itself
             state_dict_to_load = checkpoint.state_dict()

        # Remove 'module.' prefix if saved with DataParallel
        cleaned_state_dict = {}
        is_dataparallel = any(k.startswith('module.') for k in state_dict_to_load.keys())
        if is_dataparallel:
            for k, v in state_dict_to_load.items():
                name = k[7:] # remove `module.`
                cleaned_state_dict[name] = v
        else:
            cleaned_state_dict = state_dict_to_load
            
        client_model.load_state_dict(cleaned_state_dict)
        client_model.eval()
        client_models_list.append(client_model)

    if not client_models_list:
        raise ValueError(f"No client models loaded from {args.client_models_dir}. Check paths and model files.")

    teacher_ensemble_model = Ensemble(client_models_list).to(device) # Using FedISCA's Ensemble
    teacher_ensemble_model.eval()
    print(f"Teacher ensemble created with {len(client_models_list)} client models.")

    # (Optional) Validate teacher ensemble
    criterion_eval = nn.CrossEntropyLoss().to(device) # For evaluation
    print('==> Teacher Ensemble initial validation on test set ==>')
    test(teacher_ensemble_model, data_test_loader, criterion_eval, device)

    # --- Initialize Student Model ---
    print(f"Initializing student model ({StudentModelClass.__name__})...")
    student_model = StudentModelClass(num_classes=n_classes) if input_size == 224 else StudentModelClass(in_channels=n_channels, num_classes=n_classes)
    student_model = student_model.to(device)

    if args.student_pretrained and input_size == 224 and StudentModelClass == resnet18: # If ResNet18 and 224x224
        try:
            print("  Loading ImageNet pretrained weights for student ResNet18...")
            state_dict_pt = torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/resnet18-f37072fd.pth', progress=True)
            # Remove FC layer for transfer learning
            del state_dict_pt['fc.weight']
            del state_dict_pt['fc.bias']
            student_model.load_state_dict(state_dict_pt, strict=False)
        except Exception as e:
            print(f"  Could not load ImageNet pretrained weights: {e}. Training student from scratch.")
    else:
        print("  Training student model from scratch (or no pretrained option for this model/dataset).")


    # --- Initialize DENSE Generator & Synthesizer ---
    print("Initializing DENSE Generator and AdvSynthesizer...")
    dense_generator = DenseGenerator(nz=args.nz, ngf=args.ngf, img_size=input_size, nc=n_channels).to(device)
    dense_reset_model(dense_generator) # Initialize generator weights

    img_size_tuple = (n_channels, input_size, input_size)
    adv_synthesizer = AdvSynthesizer(
        teacher=teacher_ensemble_model, # The main ensemble teacher
        model_list=client_models_list,  # Individual client models for BN hooks
        student=student_model,          # The student model being trained
        generator=dense_generator,
        nz=args.nz,
        num_classes=n_classes,
        img_size=img_size_tuple,
        iterations=args.g_steps,      # Generator updates per call to gen_data
        lr_g=args.lr_g,
        synthesis_batch_size=args.synthesis_bs,
        sample_batch_size=args.kd_bs, # Batch size for student's KD training
        adv=args.adv_scale,
        bn=args.bn_scale,
        oh=args.oh_scale,
        save_dir=dense_synthetic_img_dir, # Where AdvSynthesizer saves its images
        dataset=args.dataset # For AdvSynthesizer's internal logic if any dataset-specific transforms
    )

    # --- Setup for Student Training (Knowledge Distillation) ---
    criterion_kd = KLDiv(T=args.kd_temperature).to(device)
    optimizer_student = optim.SGD(
        student_model.parameters(),
        lr=args.lr_student_init,
        momentum=args.momentum_student,
        weight_decay=args.wd_student
    )

    # --- DENSE Training Loop ---
    print(f"Starting DENSE server training for {args.kd_epochs} epochs.")
    best_student_acc = 0.0

    for epoch in range(args.start_epoch, args.kd_epochs + 1):
        adjust_learning_rate(optimizer_student, epoch, lr_init=args.lr_student_init,
                             lr_step1=args.lr_student_step1, lr_step2=args.lr_student_step2)
        print(f"\n--- DENSE Epoch {epoch}/{args.kd_epochs} ---")
        current_lr = optimizer_student.param_groups[0]['lr']
        print(f"  Student LR: {current_lr:.6f}")

        # 1. Synthesize Data (trains the DENSE generator)
        print(f"  Step 1: Synthesizing data using AdvSynthesizer (Generator training for {args.g_steps} iterations)...")
        adv_synthesizer.gen_data(cur_ep=epoch) # This call internally trains the generator

        # 2. Train Student Model on Synthesized Data
        print(f"  Step 2: Training student model via Knowledge Distillation...")
        student_model.train()
        teacher_ensemble_model.eval() # Ensure teacher is in eval for KD

        synthetic_data_loader = adv_synthesizer.get_data()
        if not synthetic_data_loader.dataset or len(synthetic_data_loader.dataset) == 0:
            print(f"  Warning: No synthetic data generated in epoch {epoch}. Skipping student KD training for this epoch.")
        else:
            kd_epoch_loss = 0.0
            num_kd_batches = 0
            for batch_idx, synthetic_images in enumerate(synthetic_data_loader):
                synthetic_images = synthetic_images.to(device)
                optimizer_student.zero_grad()
                with torch.no_grad():
                    t_outputs = teacher_ensemble_model(synthetic_images)
                s_outputs = student_model(synthetic_images)
                
                loss_kd = criterion_kd(s_outputs, t_outputs.detach())
                loss_kd.backward()
                optimizer_student.step()
                kd_epoch_loss += loss_kd.item()
                num_kd_batches += 1

                if batch_idx % args.log_interval_kd == 0:
                    print(f"    KD Batch {batch_idx}/{len(synthetic_data_loader)}, Loss: {loss_kd.item():.4f}")
            if num_kd_batches > 0:
                 print(f"  Average KD loss for epoch {epoch}: {kd_epoch_loss/num_kd_batches:.4f}")


        # 3. Evaluate Student Model
        print(f"  Step 3: Evaluating student model on test set...")
        current_student_acc = test(student_model, data_test_loader, criterion_eval, device)
        
        with open(results_file_path, 'at') as wf:
            wf.write(f'{epoch},{current_student_acc:.4f}\n')

        if current_student_acc > best_student_acc:
            best_student_acc = current_student_acc
            print(f"  Epoch {epoch}: New BEST student accuracy: {best_student_acc:.4f}")
            torch.save(student_model.state_dict(), os.path.join(exp_descr_final, f'best_student_{args.dataset}.pth'))
        
        torch.save(student_model.state_dict(), os.path.join(exp_descr_final, f'last_student_{args.dataset}.pth'))
        print(f"--- Epoch {epoch} complete. Current Acc: {current_student_acc:.4f} (Best Acc: {best_student_acc:.4f}) ---")

    print(f"\nDENSE training finished. Best student accuracy: {best_student_acc:.4f}")
    print(f"Results saved in: {exp_descr_final}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DENSE Server Aggregation and Training')

    # --- General arguments ---
    parser.add_argument('--dataset', default='bloodmnist', type=str,
                        choices=['bloodmnist', 'dermamnist', 'octmnist', 'pathmnist', 'tissuemnist',
                                 'isic2019', 'diabetic2015', 'rsna'],
                        help='Name of the dataset')
    parser.add_argument('--data_root', default='./dataset', type=str, help='Root directory for datasets')
    parser.add_argument('--client_models_dir', required=True, type=str,
                        help='Directory containing subfolders of pre-trained client models (e.g., client_0/best.pth)')
    parser.add_argument('--exp_descr_base', default="./results_dense", type=str,
                        help='Base directory for DENSE experiment results')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    parser.add_argument('--eval_batch_size', default=128, type=int, help='Batch size for evaluation')
    parser.add_argument('--heterogeneous_clients', action='store_true',
                        help='Set if client models have heterogeneous architectures (uses get_model_heter functions)')
    parser.add_argument('--student_pretrained', action='store_true',
                        help='Use ImageNet pretrained weights for ResNet18 student (if applicable)')


    # --- DENSE Generator & Synthesizer arguments ---
    parser.add_argument('--nz', default=256, type=int, help='Size of noise vector for DENSE generator')
    parser.add_argument('--ngf', default=64, type=int, help='Feature map size for DENSE generator')
    parser.add_argument('--lr_g', default=1e-3, type=float, help='Learning rate for DENSE generator')
    parser.add_argument('--g_steps', default=30, type=int,
                        help='Number of generator updates per AdvSynthesizer.gen_data() call')
    parser.add_argument('--synthesis_bs', default=128, type=int,
                        help='Batch size for DENSE image synthesis (generator input/output during its training)')
    parser.add_argument('--adv_scale', default=0.1, type=float, help='Scaling for DENSE AdvSynthesizer adversarial loss')
    parser.add_argument('--bn_scale', default=1.0, type=float, help='Scaling for DENSE AdvSynthesizer BN feature loss')
    parser.add_argument('--oh_scale', default=1.0, type=float, help='Scaling for DENSE AdvSynthesizer one-hot loss')

    # --- Student Knowledge Distillation arguments ---
    parser.add_argument('--kd_epochs', default=100, type=int, help='Number of epochs for student training via KD')
    parser.add_argument('--start_epoch', default=1, type=int, help='Epoch to start/resume training from')
    parser.add_argument('--kd_bs', default=128, type=int, help='Batch size for student KD training on synthetic data')
    parser.add_argument('--kd_temperature', default=20.0, type=float, help='Temperature for KLDiv in KD')
    parser.add_argument('--lr_student_init', default=0.001, type=float, help='Initial learning rate for student classifier')
    parser.add_argument('--lr_student_step1', default=50, type=int, help='Epoch to reduce student LR (1st decay)')
    parser.add_argument('--lr_student_step2', default=75, type=int, help='Epoch to reduce student LR (2nd decay)')
    parser.add_argument('--momentum_student', default=0.9, type=float, help='Momentum for student SGD optimizer')
    parser.add_argument('--wd_student', default=5e-4, type=float, help='Weight decay for student SGD optimizer')
    parser.add_argument('--log_interval_kd', default=10, type=int, help='Log frequency for KD batches')


    args = parser.parse_args()
    main_dense(args)