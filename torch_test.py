import torch
print(torch.__version__)                # 输出 PyTorch 版本，如 2.4.1+cu121
print(torch.cuda.is_available())        # 输出 True/False，表示 CUDA 是否可用