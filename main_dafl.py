import argparse
import os
import random
import sys # For DAFL's original sys.exit() on error

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
from utils import test as fedisca_test, Ensemble, adjust_learning_rate as fedisca_adjust_lr # FedISCA utils

# DAFL specific code imports
from dafl_code.dafl_generator_model import Generator as DaflGenerator
from dafl_code.dafl_utils import kdloss as dafl_kdloss, AvgrageMeter as DaflAvgrageMeter, accuracy as dafl_accuracy

# Note: DAFL's resnet and lenet are not directly used here for teacher/student
# We rely on FedISCA's model loading for teachers and configurable student.


def extract_teacher_features(teacher_model, input_data, opt_dafl, device):
    """
    Helper to extract features for DAFL's activation loss.
    This is a placeholder and needs careful implementation based on teacher architecture.
    For simplicity, if teacher is an ensemble, this might be hard to define generically.
    If we assume homogeneous teachers of a known type (e.g. torchvision resnet),
    we can try to apply a hook like DAFL's original code.
    """
    if not hasattr(teacher_model, 'module') and not isinstance(teacher_model, nn.DataParallel): # Not wrapped by DataParallel
        # Check if it's a single model of a type DAFL knows (e.g., torchvision ResNet)
        # This is a simplified check.
        if isinstance(teacher_model, (resnet18, ResNet18CIFAR)) and hasattr(teacher_model, 'avgpool'):
            features_out = [torch.Tensor().to(device)]
            def hook(module, input, output):
                features_out[0] = output.view(output.size(0), -1) # Flatten after avgpool
            
            handle = teacher_model.avgpool.register_forward_hook(hook)
            _ = teacher_model(input_data) # Forward pass to trigger hook
            handle.remove()
            return features_out[0]
        elif isinstance(teacher_model, Ensemble) and opt_dafl.a_dafl > 0:
            # For an ensemble, it's tricky. One might average features, or take from one member,
            # or simply not use activation loss for ensembles unless a clear strategy is defined.
            print("Warning: Activation loss with Ensemble teacher is non-trivial. Returning zero tensor for features.")
            # Fallback: return a zero tensor of expected shape or disable this loss for ensembles.
            # Let's assume first model in ensemble and it's a ResNet
            if len(teacher_model.models) > 0:
                 first_model = teacher_model.models[0]
                 if isinstance(first_model, (resnet18, ResNet18CIFAR)) and hasattr(first_model, 'avgpool'):
                     return extract_teacher_features(first_model, input_data, opt_dafl, device) # Recursive call
            
            # Placeholder: if features are needed, this needs to be robust
            # For now, let's assume the output of the forward pass of the teacher can be used
            # if avgpool is not present. This is not ideal for activation loss.
            # A dummy tensor based on batch size, assuming some feature dim
            return torch.zeros(input_data.size(0), 512, device=device) # Example: 512 for ResNet18 block
            
    elif hasattr(teacher_model, 'module') and hasattr(teacher_model.module, 'avgpool'): # Wrapped by DataParallel
        # Handle DataParallel wrapped model
        features_out = [torch.Tensor().to(device)]
        def hook(module, input, output):
            features_out[0] = output.view(output.size(0), -1)
        handle = teacher_model.module.avgpool.register_forward_hook(hook)
        _ = teacher_model(input_data)
        handle.remove()
        return features_out[0]
    
    # Fallback if no specific feature extraction is possible or for complex ensembles
    # print("Warning: Could not extract teacher features for activation loss. Returning zero tensor.")
    return torch.zeros(input_data.size(0), 1, device=device) # Return a minimal tensor to avoid errors if opt.a_dafl > 0

def main_dafl(args):
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
    exp_descr_final = os.path.join(args.exp_descr_base, f"{args.dataset}_dafl_server")
    os.makedirs(exp_descr_final, exist_ok=True)

    results_file_name = f'test_results_{args.dataset}_dafl.csv'
    results_file_path = os.path.join(exp_descr_final, results_file_name)
    # if os.path.isfile(results_file_path) and args.start_epoch == 1:
    #     os.remove(results_file_path) # Clear on new run

    # --- Load Dataset Info & Test Loader ---
    # (This section is similar to run_dense_server.py, ensure consistency)
    if args.dataset in medmnist.INFO:
        info = medmnist.INFO[args.dataset]
        DataClass = getattr(medmnist, info['python_class'])
        transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[.5], std=[.5])])
        if args.dataset == 'pathmnist': # PathMNIST is 3x224x224 but medmnist info says 28x28
            args.img_size = 224 # Override for pathmnist if needed based on actual data
            args.channels = 3
            transform_test = transforms.Compose([
                transforms.Resize(args.img_size), # Ensure it's 224 for pathmnist
                transforms.ToTensor(), 
                transforms.Normalize(mean=[.5,.5,.5], std=[.5,.5,.5])
            ])


        data_test = DataClass(split='test', transform=transform_test, download=True, root=os.path.join(args.data_root, 'medmnist'))
        data_test_loader = torch.utils.data.DataLoader(dataset=data_test, batch_size=args.eval_batch_size, shuffle=False, num_workers=8)
        
        args.img_size = 28 if args.dataset not in ['pathmnist'] else 224 # DAFL uses this
        args.channels = info['n_channels'] if args.dataset not in ['pathmnist'] else 3
        n_classes = len(info['label'])
        StudentModelClass = ResNet18CIFAR if args.img_size == 28 else resnet18
        ClientModelClass = ResNet18CIFAR if args.img_size == 28 else resnet18

    # ... (add other dataset handlers from run_dense_server.py: isic2019, diabetic2015, rsna) ...
    elif args.dataset == 'isic2019':
        from dataset_isic2019 import FedIsic2019
        args.img_size = 224; args.channels = 3; n_classes = 8
        transform_test = transforms.Compose([transforms.CenterCrop(args.img_size), transforms.ToTensor(), transforms.Normalize(mean=[.5]*3, std=[.5]*3)])
        data_test_loader = torch.utils.data.DataLoader(
            FedIsic2019(split='test', data_path=os.path.join(args.data_root, 'fed_isic2019'), transform=transform_test),
            batch_size=args.eval_batch_size, shuffle=False, num_workers=8)
        StudentModelClass = resnet18; ClientModelClass = resnet18
    elif args.dataset == 'diabetic2015':
        from torchvision.datasets import ImageFolder
        args.img_size = 224; args.channels = 3; n_classes = 5
        transform_test = transforms.Compose([transforms.CenterCrop(args.img_size), transforms.ToTensor(), transforms.Normalize(mean=[.5]*3, std=[.5]*3)])
        data_test_loader = torch.utils.data.DataLoader(
            ImageFolder(os.path.join(args.data_root, 'diabetic2015', 'test'), transform=transform_test),
            batch_size=args.eval_batch_size, shuffle=False, num_workers=8)
        StudentModelClass = resnet18; ClientModelClass = resnet18
    elif args.dataset == 'rsna':
        from torchvision.datasets import ImageFolder
        args.img_size = 224; args.channels = 3; n_classes = 2
        transform_test = transforms.Compose([transforms.CenterCrop(args.img_size), transforms.ToTensor(), transforms.Normalize(mean=[.5]*3, std=[.5]*3)])
        data_test_loader = torch.utils.data.DataLoader(
            ImageFolder(os.path.join(args.data_root, 'rsna', 'test'), transform=transform_test),
            batch_size=args.eval_batch_size, shuffle=False, num_workers=8)
        StudentModelClass = resnet18; ClientModelClass = resnet18
    else:
        # Fallback or use DAFL's CIFAR/MNIST if needed, but FedISCA focuses on medical
        raise ValueError(f"Dataset {args.dataset} not fully configured for DAFL in FedISCA. Please add specific handling.")

    print(f"DAFL for dataset: {args.dataset}, img_size: {args.img_size}, channels: {args.channels}, classes: {n_classes}")

    # --- Load Client Models (Teachers) & Create Ensemble ---
    client_models_list = []
    # ... (Loading logic identical to run_dense_server.py) ...
    if not os.path.isdir(args.client_models_dir):
        raise ValueError(f"Client models directory not found: {args.client_models_dir}")

    print(f"Loading client models from: {args.client_models_dir}")
    for client_folder_name in sorted(os.listdir(args.client_models_dir)):
        client_model_path = os.path.join(args.client_models_dir, client_folder_name, 'best.pth')
        if not os.path.isfile(client_model_path): continue

        client_id_str = client_folder_name.split('_')[-1]
        if not client_id_str.isdigit(): continue
        client_id = int(client_id_str)

        current_input_channels = args.channels
        current_num_classes = n_classes

        if args.heterogeneous_clients:
            get_heter_model_func = get_model_heter_224 if args.img_size == 224 else get_model_heter
            client_model = get_heter_model_func(client_id, in_channels=current_input_channels, num_classes=current_num_classes)
        else:
            client_model = ClientModelClass(num_classes=current_num_classes) if args.img_size == 224 else ClientModelClass(in_channels=current_input_channels, num_classes=current_num_classes)
        
        client_model = client_model.to(device)
        checkpoint = torch.load(client_model_path, map_location=device)
        state_dict_to_load = checkpoint.get('state_dict', checkpoint.get('model_state_dict', checkpoint)) # common patterns
        if hasattr(state_dict_to_load, 'state_dict'): state_dict_to_load = state_dict_to_load.state_dict()

        cleaned_state_dict = {}
        is_dataparallel = any(k.startswith('module.') for k in state_dict_to_load.keys())
        for k, v in state_dict_to_load.items():
            name = k[7:] if is_dataparallel else k
            cleaned_state_dict[name] = v
            
        client_model.load_state_dict(cleaned_state_dict)
        client_model.eval()
        client_models_list.append(client_model)
    
    if not client_models_list: raise ValueError(f"No client models loaded from {args.client_models_dir}.")
    teacher_ensemble_model = Ensemble(client_models_list).to(device)
    teacher_ensemble_model.eval()
    print(f"Teacher ensemble created with {len(client_models_list)} client models.")
    criterion_eval = nn.CrossEntropyLoss().to(device)
    fedisca_test(teacher_ensemble_model, data_test_loader, criterion_eval, device)


    # --- Initialize Student Model ---
    student_model = StudentModelClass(num_classes=n_classes) if args.img_size == 224 else StudentModelClass(in_channels=args.channels, num_classes=n_classes)
    student_model = student_model.to(device)
    # ... (Optional: Load pretrained weights for student, similar to run_dense_server.py) ...
    if args.student_pretrained and args.img_size == 224 and StudentModelClass == resnet18:
        try:
            print("  Loading ImageNet pretrained weights for student ResNet18...")
            state_dict_pt = torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/resnet18-f37072fd.pth', progress=True)
            del state_dict_pt['fc.weight']; del state_dict_pt['fc.bias']
            student_model.load_state_dict(state_dict_pt, strict=False)
        except Exception as e:
            print(f"  Could not load ImageNet pretrained weights: {e}. Student from scratch.")


    # --- Initialize DAFL Generator ---
    # DAFL Generator expects 'opt' object with img_size, latent_dim, channels
    dafl_gen_opt = argparse.Namespace(img_size=args.img_size, latent_dim=args.latent_dim_dafl, channels=args.channels)
    dafl_generator = DaflGenerator(dafl_gen_opt).to(device)
    # Note: DAFL's original code doesn't explicitly reset generator weights, but it's good practice.
    # We can add a simple reinitialization if needed, or assume it starts fresh.

    # --- Optimizers for DAFL (Generator and Student) ---
    optimizer_G = optim.Adam(dafl_generator.parameters(), lr=args.lr_g_dafl)
    optimizer_S = optim.Adam(student_model.parameters(), lr=args.lr_s_dafl) # DAFL uses Adam for student too on MNIST
    # Or use SGD for student like FedISCA/DENSE, adjust LR if so:
    # optimizer_S = optim.SGD(student_model.parameters(), lr=args.lr_s_dafl, momentum=0.9, weight_decay=5e-4)


    # --- DAFL Co-Training Loop ---
    print(f"Starting DAFL server co-training for {args.dafl_epochs} epochs.")
    best_student_acc = 0.0
    criterion_ce_for_generator = nn.CrossEntropyLoss().to(device) # For DAFL's one-hot loss

    for epoch in range(args.start_epoch, args.dafl_epochs + 1):
        # DAFL's original lr_S is sometimes adjusted by epoch (e.g. for CIFAR),
        # FedISCA's adjust_learning_rate can be used if optimizer_S is SGD.
        # If Adam for student, typically fixed LR or use Adam's own beta parameters.
        # For simplicity, using fixed LR for Adam here, or FedISCA's scheduler if SGD.
        # fedisca_adjust_lr(optimizer_S, epoch, lr_init=args.lr_s_dafl,
        #                    lr_step1=args.lr_s_step1, lr_step2=args.lr_s_step2)


        # Inner loop for co-training steps (DAFL uses a fixed number of iterations, e.g., 120)
        # Let's make this configurable, e.g., args.dafl_inner_steps
        dafl_generator.train()
        student_model.train()
        teacher_ensemble_model.eval() # Teacher always in eval

        avg_loss_g_one_hot = DaflAvgrageMeter()
        avg_loss_g_info_entropy = DaflAvgrageMeter()
        avg_loss_g_activation = DaflAvgrageMeter()
        avg_loss_s_kd = DaflAvgrageMeter()
        
        for inner_step in range(args.dafl_inner_steps):
            z = torch.randn(args.dafl_batch_size, args.latent_dim_dafl).to(device)
            
            optimizer_G.zero_grad()
            optimizer_S.zero_grad()

            gen_imgs = dafl_generator(z)
            
            # Teacher outputs (no grad for teacher)
            with torch.no_grad():
                outputs_T = teacher_ensemble_model(gen_imgs)
            
            # --- Generator Losses (DAFL specific) ---
            # 1. One-Hot Loss (Encourage confident & consistent teacher predictions)
            pred_T = outputs_T.data.max(1)[1] # Pseudo-labels from teacher
            loss_g_one_hot = criterion_ce_for_generator(outputs_T, pred_T) * args.oh_dafl

            # 2. Information Entropy Loss (Encourage class diversity)
            softmax_o_T = F.softmax(outputs_T, dim=1).mean(dim=0) # Avg softmax over batch
            loss_g_info_entropy = (softmax_o_T * torch.log10(softmax_o_T + 1e-8)).sum() * args.ie_dafl # Add epsilon for stability

            # 3. Activation Loss (Encourage strong features from teacher)
            loss_g_activation = torch.tensor(0.0, device=device)
            if args.a_dafl > 0:
                # This requires a robust way to get features_T from teacher_ensemble_model
                # For now, this part is simplified. See extract_teacher_features.
                features_T = extract_teacher_features(teacher_ensemble_model, gen_imgs, args, device)
                if features_T is not None and features_T.numel() > 0 : # Check if features were extracted
                     loss_g_activation = -features_T.abs().mean() * args.a_dafl
                else: # If features_T is None or empty, skip this loss
                    if args.a_dafl > 0: print(f"Epoch {epoch}, Step {inner_step}: Skipping activation loss as features_T is None/empty.")
                    loss_g_activation = torch.tensor(0.0, device=device)


            # --- Student KD Loss ---
            # Student learns from teacher on these *same* generated images
            # Detach gen_imgs for student if generator also backprops through student (not in DAFL original loss formulation for G)
            # Detach outputs_T for student loss calculation
            outputs_S = student_model(gen_imgs.detach()) # Student sees detached images
            loss_s_kd = dafl_kdloss(outputs_S, outputs_T.detach())

            # --- Total Loss & Backpropagation ---
            # DAFL's loss combines G losses and S_KD loss to update G and S *simultaneously*
            total_dafl_loss = loss_g_one_hot + loss_g_info_entropy + loss_g_activation + loss_s_kd
            
            total_dafl_loss.backward()
            optimizer_G.step()
            optimizer_S.step()

            avg_loss_g_one_hot.update(loss_g_one_hot.item() / args.oh_dafl if args.oh_dafl > 0 else 0, gen_imgs.size(0))
            avg_loss_g_info_entropy.update(loss_g_info_entropy.item() / args.ie_dafl if args.ie_dafl > 0 else 0, gen_imgs.size(0))
            avg_loss_g_activation.update(loss_g_activation.item() / args.a_dafl if args.a_dafl > 0 else 0, gen_imgs.size(0))
            avg_loss_s_kd.update(loss_s_kd.item(), gen_imgs.size(0))

            if inner_step % args.log_interval_dafl == 0:
                print(f"  DAFL Epoch {epoch}, InnerStep {inner_step}/{args.dafl_inner_steps}: "
                      f"G_OH: {loss_g_one_hot.item():.3f}, G_IE: {loss_g_info_entropy.item():.3f}, "
                      f"G_Act: {loss_g_activation.item():.3f}, S_KD: {loss_s_kd.item():.3f}")
        
        print(f"  DAFL Epoch {epoch} Avg Losses: "
              f"G_OH: {avg_loss_g_one_hot.avg:.3f}, G_IE: {avg_loss_g_info_entropy.avg:.3f}, "
              f"G_Act: {avg_loss_g_activation.avg:.3f}, S_KD: {avg_loss_s_kd.avg:.3f}")

        # Evaluate Student Model after each DAFL epoch
        print(f"  Evaluating student model on test set...")
        current_student_acc = fedisca_test(student_model, data_test_loader, criterion_eval, device)
        
        with open(results_file_path, 'at') as wf:
            wf.write(f'{epoch},{current_student_acc:.4f}\n')

        if current_student_acc > best_student_acc:
            best_student_acc = current_student_acc
            print(f"  Epoch {epoch}: New BEST student accuracy: {best_student_acc:.4f}")
            torch.save(student_model.state_dict(), os.path.join(exp_descr_final, f'best_student_{args.dataset}.pth'))
        
        torch.save(student_model.state_dict(), os.path.join(exp_descr_final, f'last_student_{args.dataset}.pth'))
        # Optionally save generator
        # torch.save(dafl_generator.state_dict(), os.path.join(exp_descr_final, f'last_generator_{args.dataset}.pth'))

        print(f"--- DAFL Epoch {epoch} complete. Current Acc: {current_student_acc:.4f} (Best Acc: {best_student_acc:.4f}) ---")

    print(f"\nDAFL training finished. Best student accuracy: {best_student_acc:.4f}")
    print(f"Results saved in: {exp_descr_final}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DAFL Server Aggregation and Co-Training')

    # --- General arguments (similar to run_dense_server.py) ---
    parser.add_argument('--dataset', default='bloodmnist', type=str,
                        choices=['bloodmnist', 'dermamnist', 'octmnist', 'pathmnist', 'tissuemnist',
                                 'isic2019', 'diabetic2015', 'rsna', 'MNIST', 'cifar10', 'cifar100'], # Added DAFL originals
                        help='Name of the dataset')
    parser.add_argument('--data_root', default='./dataset', type=str, help='Root directory for datasets')
    parser.add_argument('--client_models_dir', required=True, type=str,
                        help='Directory containing subfolders of pre-trained client models')
    parser.add_argument('--exp_descr_base', default="./results_dafl", type=str,
                        help='Base directory for DAFL experiment results')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    parser.add_argument('--eval_batch_size', default=128, type=int, help='Batch size for evaluation')
    parser.add_argument('--heterogeneous_clients', action='store_true',
                        help='Set if client models have heterogeneous architectures')
    parser.add_argument('--student_pretrained', action='store_true',
                        help='Use ImageNet pretrained weights for ResNet18 student (if applicable)')
    # img_size and channels will be set based on dataset, but DAFL's Generator needs them.
    # These are added to args by the dataset loading logic now.


    # --- DAFL Specific Arguments (from DAFL_train.py) ---
    parser.add_argument('--dafl_epochs', default=200, type=int, help='Number of epochs for DAFL co-training')
    parser.add_argument('--start_epoch', default=1, type=int, help='Epoch to start/resume DAFL training from')
    parser.add_argument('--dafl_batch_size', default=512, type=int, help='Batch size for generating images and DAFL losses')
    parser.add_argument('--lr_g_dafl', default=0.002, type=float, help='Learning rate for DAFL Generator (DAFL used 0.2, often too high for Adam, 0.002 is safer start for Adam)') # DAFL original was 0.2, but for Adam typically smaller
    parser.add_argument('--lr_s_dafl', default=2e-3, type=float, help='Learning rate for DAFL Student')
    # LR scheduler for student (if using SGD, these map to fedisca_adjust_lr)
    # parser.add_argument('--lr_s_step1', default=80, type=int) # Example if using SGD and FedISCA's scheduler
    # parser.add_argument('--lr_s_step2', default=120, type=int)

    parser.add_argument('--latent_dim_dafl', default=100, type=int, help='Dimensionality of the latent space for DAFL Generator')
    parser.add_argument('--oh_dafl', type=float, default=1.0, help='Weight for DAFL one-hot loss')
    parser.add_argument('--ie_dafl', type=float, default=0.1, help='Weight for DAFL information entropy loss (DAFL original: 5, but can be unstable, try smaller)')
    parser.add_argument('--a_dafl', type=float, default=0.0, help='Weight for DAFL activation loss (default 0 to make it optional due to complexity with ensembles)') # Default 0.1 in DAFL original for some cases

    parser.add_argument('--dafl_inner_steps', default=120, type=int, help='Number of co-training steps per DAFL epoch')
    parser.add_argument('--log_interval_dafl', default=20, type=int, help='Log frequency for DAFL inner steps')

    # These will be determined by dataset choice and set into args inside main_dafl
    # parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension - set by dataset')
    # parser.add_argument('--channels', type=int, default=3, help='number of image channels - set by dataset')


    args = parser.parse_args()
    main_dafl(args)