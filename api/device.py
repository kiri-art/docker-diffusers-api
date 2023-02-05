import torch

if torch.cuda.is_available():
    print("[device] CUDA (Nvidia) detected")
    device_id = "cuda"
    device_name = torch.cuda.get_device_name()
elif torch.backends.mps.is_available():
    print("[device] MPS (MacOS Metal, Apple M1, etc) detected")
    device_id = "mps"
    device_name = "MPS"
else:
    print("[device] CPU only - no GPU detected")
    device_id = "cpu"
    device_name = "CPU only"

    if not torch.backends.cuda.is_built():
        print(
            "CUDA not available because the current PyTorch install was not "
            "built with CUDA enabled."
        )
    if torch.backends.mps.is_built():
        print(
            "MPS not available because the current MacOS version is not 12.3+ "
            "and/or you do not have an MPS-enabled device on this machine."
        )
    else:
        print(
            "MPS not available because the current PyTorch install was not "
            "built with MPS enabled."
        )

device = torch.device(device_id)
