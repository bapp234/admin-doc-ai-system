import sys
import os
import torch
from loguru import logger

def optimize_hardware() -> bool:
    threads = "4"
    os.environ["OMP_NUM_THREADS"] = threads
    os.environ["OPENBLAS_NUM_THREADS"] = threads
    os.environ["MKL_NUM_THREADS"] = threads
    os.environ["VECLIB_MAXIMUM_THREADS"] = threads
    os.environ["NUMEXPR_NUM_THREADS"] = threads
    torch.set_num_threads(int(threads))
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        torch.cuda.empty_cache()
        # Enable cuDNN autotuner for better performance on fixed-size inputs
        torch.backends.cudnn.benchmark = True
        # Prefer performance over strict determinism for throughput
        torch.backends.cudnn.deterministic = False
        logger.info(f"Using GPU: {device_name}")
        return True
    else:
        return False