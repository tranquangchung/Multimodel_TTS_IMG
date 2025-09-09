import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import torch.nn as nn
from torch.utils.data import DataLoader


# Khởi tạo WarmupLR scheduler
class WarmupLR(_LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 25000,
        max_lr: float = 0.002,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        super().__init__(optimizer, last_epoch)

    def __repr__(self):
        return f"{self.__class__.__name__}(warmup_steps={self.warmup_steps}, max_lr={self.max_lr})"

    def get_lr(self):
        step_num = self.last_epoch + 1
        if step_num < self.warmup_steps:
            # Linear warmup from 0 to max_lr
            return [base_lr * (step_num / self.warmup_steps) for base_lr in self.base_lrs]
        else:
            # After warmup, decay the learning rate (or keep it constant)
            return [self.max_lr for _ in self.base_lrs]

    def set_step(self, step: int):
        self.last_epoch = step


# Khởi tạo optimizer
def init_optimizer(model, lr=0.002, optimizer_type='adam'):
    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    return optimizer


# Hàm hiển thị learning rate tại mỗi epoch
def print_lr(epoch, optimizer, scheduler=None):
    lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch + 1}: Learning rate = {lr:.8f}")
    if scheduler is not None:
        print(f"Epoch {epoch + 1}: Learning rate after scheduler step = {scheduler.get_lr()[0]:.8f}")


# Mô hình đơn giản để thử nghiệm
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)


# Tạo DataLoader giả để kiểm tra
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, size=100):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Trả về một tensor ngẫu nhiên và target giả
        return torch.randn(10), torch.tensor(0)


# Hàm huấn luyện qua nhiều epoch và theo dõi learning rate
def train_epoch(model, train_loader, optimizer, scheduler=None, grad_clip=5.0):
    model.train()
    for batch_idx, (data, _) in enumerate(train_loader):
        optimizer.zero_grad()  # Reset gradients
        # output = model(data)  # Forward pass (chỉ tính output, không cần tính loss)

        # Giả sử loss là một giá trị random (chỉ để minh họa)
        # loss = torch.randn(1)  # Đây chỉ là ví dụ, trong thực tế bạn sẽ tính loss tại đây
        # loss.backward()  # Backpropagate

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()  # Cập nhật tham số của model

    if scheduler is not None:
        scheduler.step()  # Cập nhật learning rate theo scheduler


# Main để theo dõi learning rate
def main():
    # Khởi tạo model, optimizer và scheduler
    model = SimpleModel()
    train_loader = DataLoader(DummyDataset(), batch_size=16, shuffle=True)

    # Cấu hình optimizer và scheduler
    optimizer = init_optimizer(model, lr=0.002, optimizer_type='adam')
    scheduler = WarmupLR(optimizer, warmup_steps=25000, max_lr=0.002)

    # Các tham số huấn luyện
    max_epoch = 100
    grad_clip = 5.0
    log_interval = 100

    # Huấn luyện qua nhiều epoch và theo dõi learning rate
    for epoch in range(max_epoch):
        # Huấn luyện qua từng batch
        train_epoch(model, train_loader, optimizer, scheduler, grad_clip)

        # In learning rate mỗi epoch
        print_lr(epoch, optimizer, scheduler)

        # Log mỗi log_interval (hoặc mỗi epoch)
        if (epoch + 1) % log_interval == 0:
            print(f"Epoch {epoch + 1} completed.")


if __name__ == "__main__":
    main()
