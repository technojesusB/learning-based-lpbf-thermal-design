import torch

def cuda_check(print_out: bool = False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if print_out:
        print("CUDA available:", torch.cuda.is_available())
        print(f"Device: {device} (PyTorch Version: {torch.__version__})")
        if device == "cuda":
            print(f"GPU name: {torch.cuda.get_device_name(0)}")
            
    return device