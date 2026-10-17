import sys
import torch

print("当前 Python：", sys.executable)
print("PyTorch 版本：", torch.__version__)

x = torch.ones(2, 3)
print("张量：")
print(x)
print("运算结果：", x.sum().item())