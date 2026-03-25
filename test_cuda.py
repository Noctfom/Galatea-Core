import torch
print("尝试连接显卡...")
free, total = torch.cuda.mem_get_info()
print(f"显存可用: {free / 1024**3:.2f} GB / {total / 1024**3:.2f} GB")
a = torch.randn(10, device='cuda')
print("✅ 显卡正常运作！")