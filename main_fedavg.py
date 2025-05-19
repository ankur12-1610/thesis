# File: aggregate_fedisca_teachers.py
# Description: Standalone script to perform FedAvg aggregation on trained
#              teacher/client models from the FedISCA project.
import torch
import os
import argparse
import collections # For checking state_dict type from loaded checkpoint

# Assuming this script is run from the root of the FedISCA project,
# or the 'models' and 'utils' modules are otherwise in PYTHONPATH.
from models import get_model_heter, get_model_heter_224
# from models.resnet_cifar import ResNet18 # Example, if used directly by get_model_heter
from utils import average_weights # From FedISCA's utils.py

def load_single_teacher_model(model_arch_index,
                              weight_path,
                              device,
                              input_size_type, # '28' or '224'
                              num_classes,
                              in_channels):
    """
    Instantiates and loads weights for a single teacher model.
    For FedAvg, all models aggregated should resolve to the same architecture
    by using the same model_arch_index and other parameters.
    """
    model_instance = None
    is_224_input = (input_size_type == '224')

    _get_model_fn = get_model_heter_224 if is_224_input else get_model_heter
    
    try:
        # print(f"Instantiating model: arch_index={model_arch_index}, nc={num_classes}, ich={in_channels}, 224px={is_224_input}")
        model_instance = _get_model_fn(model_arch_index, num_classes=num_classes, in_channels=in_channels)
    except Exception as e:
        print(f"Error: Could not instantiate model with arch_index {model_arch_index}: {e}")
        return None

    model_instance = model_instance.to(device) # Move to device before loading state_dict
    
    try:
        # print(f"Loading weights from: {weight_path}")
        # Load checkpoint, assuming it's a state_dict or a model object
        checkpoint = torch.load(weight_path, map_location=device)
        
        if isinstance(checkpoint, collections.OrderedDict): # Most common: state_dict saved
            model_instance.load_state_dict(checkpoint)
        elif hasattr(checkpoint, 'state_dict'): # Model object saved
            model_instance.load_state_dict(checkpoint.state_dict())
        else: # Fallback for other types, though less common for .pth from train_classifier.py
            model_instance.load_state_dict(checkpoint)
            
        model_instance.eval() # Set to evaluation mode
        # print(f"Successfully loaded weights from {weight_path}")
        return model_instance
    except FileNotFoundError:
        print(f"Error: Weight file not found at {weight_path}.")
        return None
    except RuntimeError as e:
        print(f"Error: RuntimeError loading state_dict from {weight_path}. "
              f"This often indicates a mismatch between the instantiated model architecture "
              f"(arch_index {model_arch_index}) and the saved weights. Error: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred loading model from {weight_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description="FedAvg Aggregation Script for FedISCA Trained Teacher Models.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--models_input_dir', type=str, required=True,
        help="Directory containing subdirectories for each client (e.g., client_0, client_1), \n"
             "where each client subdirectory is expected to contain a 'best.pth' model file \n"
             "(as saved by FedISCA's train_classifier.py)."
    )
    parser.add_argument(
        '--aggregated_model_output_path', type=str, required=True,
        help="Full path (including filename, e.g., ./fedavg_aggregated_teacher.pth) to save the \n"
             "resulting aggregated model's state_dict."
    )
    parser.add_argument(
        '--model_arch_index', type=int, required=True,
        help="Integer index specifying the model architecture to load for ALL clients. \n"
             "This corresponds to the 'index' argument for get_model_heter or get_model_heter_224. \n"
             "This ensures all models loaded for aggregation are homogeneous."
    )
    parser.add_argument(
        '--input_size_type', type=str, required=True, choices=['28', '224'],
        help="Input size type for the models: '28' (e.g., MedMNIST) or '224' (e.g., ISIC, RSNA)."
    )
    parser.add_argument(
        '--in_channels', type=int, required=True,
        help="Number of input channels for the models (e.g., 1 for MNIST, 3 for RGB images)."
    )
    parser.add_argument(
        '--num_classes', type=int, required=True,
        help="Number of output classes for the models."
    )
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
        choices=['cuda', 'cpu'], help="Device to use for loading models. Aggregation itself is CPU-based."
    )

    args = parser.parse_args()
    print(f"Running FedAvg aggregation for FedISCA teachers with arguments: {args}")

    current_device = torch.device(args.device)
    print(f"Using device: {current_device} for model loading.")

    loaded_models_list = [] # List to store loaded model objects
    
    if not os.path.isdir(args.models_input_dir):
        print(f"Error: Specified models input directory '{args.models_input_dir}' does not exist.")
        return

    # train_classifier.py saves models in subdirs like 'client_0', 'client_1'
    # These subdirs are typically inside a path like 'pretrained_models/dataset_partition_users_beta/'
    # So, models_input_dir should point to this 'dataset_partition_users_beta' like directory.
    client_subdirs = sorted([
        d for d in os.listdir(args.models_input_dir)
        if os.path.isdir(os.path.join(args.models_input_dir, d)) and d.startswith('client_')
    ])

    if not client_subdirs:
        print(f"No client subdirectories (e.g., 'client_0', 'client_1', ...) found in '{args.models_input_dir}'.")
        print("Please ensure --models_input_dir points to the directory containing these client-specific folders.")
        return

    print(f"Found {len(client_subdirs)} potential client directories. Attempting to load models with architecture index {args.model_arch_index}...")

    for client_dirname in client_subdirs:
        client_dir_path = os.path.join(args.models_input_dir, client_dirname)
        weight_file_name = 'best.pth' # As saved by train_classifier.py
        weight_path = os.path.join(client_dir_path, weight_file_name)

        if os.path.isfile(weight_path):
            # print(f"Attempting to load model for client '{client_dirname}' from '{weight_path}'...")
            model = load_single_teacher_model(
                model_arch_index=args.model_arch_index, # Enforces homogeneity
                weight_path=weight_path,
                device=current_device,
                input_size_type=args.input_size_type,
                num_classes=args.num_classes,
                in_channels=args.in_channels
            )
            if model:
                loaded_models_list.append(model)
                print(f"Successfully loaded model from: {client_dir_path}/{weight_file_name}")
        else:
            print(f"Weight file '{weight_file_name}' not found in '{client_dir_path}'. Skipping this client.")

    if not loaded_models_list:
        print("No models were successfully loaded. FedAvg aggregation cannot proceed.")
        return

    print(f"\nSuccessfully loaded {len(loaded_models_list)} client models for aggregation.")
    
    # Extract state_dicts and move to CPU for average_weights function
    # (as average_weights uses copy.deepcopy which is safer with CPU tensors)
    client_state_dicts_on_cpu = []
    for model_obj in loaded_models_list:
        client_state_dicts_on_cpu.append(model_obj.to('cpu').state_dict())

    print(f"Starting FedAvg parameter aggregation using 'average_weights' from utils.py...")
    try:
        aggregated_state_dict = average_weights(client_state_dicts_on_cpu)
        print("FedAvg aggregation successful.")
    except ValueError as ve: # Catch specific errors from average_weights if keys/shapes mismatch
        print(f"Error during FedAvg aggregation: {ve}")
        print("This indicates that despite loading with the same arch_index, the underlying state_dicts are incompatible.")
        return
    except Exception as e:
        print(f"An unexpected error occurred during FedAvg aggregation: {e}")
        return

    # Ensure output directory exists before saving
    output_dir = os.path.dirname(args.aggregated_model_output_path)
    if output_dir and not os.path.exists(output_dir): # If output_path includes a directory part
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")
        except OSError as e:
            print(f"Error creating output directory '{output_dir}': {e}. Please check permissions and path.")
            return
    
    try:
        torch.save(aggregated_state_dict, args.aggregated_model_output_path)
        print(f"Aggregated FedAvg model state_dict saved to: {args.aggregated_model_output_path}")
        print("\nTo use this aggregated model in main_fedisca.py, point the --teacher_weights argument to this .pth file.")
    except Exception as e:
        print(f"Error saving the aggregated model state_dict to '{args.aggregated_model_output_path}': {e}")

if __name__ == '__main__':
    main()
