import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import random
import sys
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from models.cyclegan import ResNetGenerator, Discriminator


class ImageBuffer:
    """Stores recently generated images to reduce training oscillation."""
    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        self.buffer = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        result = []
        for img in images.split(1, dim=0):
            if len(self.buffer) < self.pool_size:
                self.buffer.append(img)
                result.append(img)
            elif random.random() > 0.5:
                idx = random.randint(0, len(self.buffer) - 1)
                old_img = self.buffer[idx].clone()
                self.buffer[idx] = img
                result.append(old_img)
            else:
                result.append(img)
        return torch.cat(result, dim=0)


class UnpairedDataset(Dataset):
    """Dataset for unpaired image-to-image translation."""
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.dir_A = os.path.join(root_dir, 'NORMAL')
        self.dir_B = os.path.join(root_dir, 'PNEUMONIA')
        self.files_A = [os.path.join(self.dir_A, f) for f in os.listdir(self.dir_A)
                        if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
        self.files_B = [os.path.join(self.dir_B, f) for f in os.listdir(self.dir_B)
                        if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
        self.size_A = len(self.files_A)
        self.size_B = len(self.files_B)

    def __len__(self):
        return max(self.size_A, self.size_B)

    def __getitem__(self, index):
        img_A = Image.open(self.files_A[index % self.size_A]).convert('RGB')
        img_B = Image.open(self.files_B[random.randint(0, self.size_B - 1)]).convert('RGB')
        if self.transform:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)
        return {'A': img_A, 'B': img_B}


def get_lr_lambda(n_epochs, decay_start_epoch):
    """Linear LR decay starting from decay_start_epoch."""
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch - decay_start_epoch) / float(n_epochs - decay_start_epoch + 1)
        return lr_l
    return lambda_rule


def train():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',   type=str,   default='data/chest_xray/train')
    parser.add_argument('--epochs',     type=int,   default=20, 
                        help='Total epochs (LR decays linearly after epoch 10)')
    parser.add_argument('--decay_epoch',type=int,   default=10,
                        help='Epoch from which LR starts decaying')
    parser.add_argument('--batch_size', type=int,   default=1)
    parser.add_argument('--lr',         type=float, default=0.0002)
    parser.add_argument('--save_dir',   type=str,   default='src/models/cyclegan')
    parser.add_argument('--save_every', type=int,   default=5,
                        help='Save a checkpoint every N epochs')
    parser.add_argument('--resume',     type=int,   default=0,
                        help='Epoch number to resume from (0 = start fresh)')
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training plan: {args.epochs} total epochs, LR decay starts at epoch {args.decay_epoch}")
    print(f"Checkpoints saved every {args.save_every} epochs to '{args.save_dir}/'")

    # ── Models ─────────────────────────────────────────────────────────────────
    G_AB = ResNetGenerator(n_blocks=6).to(device)   # Normal  -> Pneumonia
    G_BA = ResNetGenerator(n_blocks=6).to(device)   # Pneumonia -> Normal
    D_A  = Discriminator().to(device)
    D_B  = Discriminator().to(device)

    if args.resume > 0:
        G_AB.load_state_dict(torch.load(os.path.join(args.save_dir, f'G_AB_epoch{args.resume}.pth'), map_location=device))
        G_BA.load_state_dict(torch.load(os.path.join(args.save_dir, f'G_BA_epoch{args.resume}.pth'), map_location=device))
        D_A.load_state_dict(torch.load(os.path.join(args.save_dir, f'D_A_epoch{args.resume}.pth'), map_location=device))
        D_B.load_state_dict(torch.load(os.path.join(args.save_dir, f'D_B_epoch{args.resume}.pth'), map_location=device))
        print(f"Resumed from epoch {args.resume}")

    # ── Losses ──────────────────────────────────────────────────────────────────
    criterion_GAN      = nn.MSELoss()
    criterion_cycle    = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    # ── Optimisers + LR schedulers ──────────────────────────────────────────────
    opt_G   = optim.Adam(list(G_AB.parameters()) + list(G_BA.parameters()), lr=args.lr, betas=(0.5, 0.999))
    opt_D_A = optim.Adam(D_A.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_D_B = optim.Adam(D_B.parameters(), lr=args.lr, betas=(0.5, 0.999))

    lr_scheduler_G   = optim.lr_scheduler.LambdaLR(opt_G,   lr_lambda=get_lr_lambda(args.epochs, args.decay_epoch))
    lr_scheduler_D_A = optim.lr_scheduler.LambdaLR(opt_D_A, lr_lambda=get_lr_lambda(args.epochs, args.decay_epoch))
    lr_scheduler_D_B = optim.lr_scheduler.LambdaLR(opt_D_B, lr_lambda=get_lr_lambda(args.epochs, args.decay_epoch))

    # Skip epochs we've already trained
    for _ in range(args.resume):
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

    # ── Image buffers ──────────────────────────────────────────────────────────
    fake_A_buffer = ImageBuffer()
    fake_B_buffer = ImageBuffer()

    # ── Data ───────────────────────────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset    = UnpairedDataset(args.data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Dataset size: {len(dataset)} items per epoch")

    # ── Training loop ──────────────────────────────────────────────────────────
    for epoch in range(args.resume, args.epochs):
        for i, batch in enumerate(dataloader):
            real_A = batch['A'].to(device)
            real_B = batch['B'].to(device)

            # ── Generators ─────────────────────────────────────────────────────
            opt_G.zero_grad()

            # Identity losses
            loss_id_A = criterion_identity(G_BA(real_A), real_A) * 5.0
            loss_id_B = criterion_identity(G_AB(real_B), real_B) * 5.0

            # GAN losses
            fake_B     = G_AB(real_A)
            loss_G_AB  = criterion_GAN(D_B(fake_B), torch.ones_like(D_B(fake_B)))
            fake_A     = G_BA(real_B)
            loss_G_BA  = criterion_GAN(D_A(fake_A), torch.ones_like(D_A(fake_A)))

            # Cycle-consistency losses
            rec_A      = G_BA(fake_B)
            loss_cyc_A = criterion_cycle(rec_A, real_A) * 10.0
            rec_B      = G_AB(fake_A)
            loss_cyc_B = criterion_cycle(rec_B, real_B) * 10.0

            loss_G = loss_G_AB + loss_G_BA + loss_cyc_A + loss_cyc_B + loss_id_A + loss_id_B
            loss_G.backward()
            opt_G.step()

            # ── Discriminator A ────────────────────────────────────────────────
            opt_D_A.zero_grad()
            fake_A_  = fake_A_buffer.query(fake_A.detach())
            loss_D_A = 0.5 * (criterion_GAN(D_A(real_A), torch.ones_like(D_A(real_A))) +
                               criterion_GAN(D_A(fake_A_), torch.zeros_like(D_A(fake_A_))))
            loss_D_A.backward()
            opt_D_A.step()

            # ── Discriminator B ────────────────────────────────────────────────
            opt_D_B.zero_grad()
            fake_B_  = fake_B_buffer.query(fake_B.detach())
            loss_D_B = 0.5 * (criterion_GAN(D_B(real_B), torch.ones_like(D_B(real_B))) +
                               criterion_GAN(D_B(fake_B_), torch.zeros_like(D_B(fake_B_))))
            loss_D_B.backward()
            opt_D_B.step()

            if i % 200 == 0:
                print(f"[Epoch {epoch+1}/{args.epochs}] [Batch {i}/{len(dataloader)}] "
                      f"[G: {loss_G.item():.3f}] [D_A: {loss_D_A.item():.3f}] [D_B: {loss_D_B.item():.3f}]")

        # Update LR schedulers
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()
        print(f"  → Epoch {epoch+1} done. LR = {lr_scheduler_G.get_last_lr()[0]:.6f}")

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            torch.save(G_AB.state_dict(), os.path.join(args.save_dir, f'G_AB_epoch{epoch+1}.pth'))
            torch.save(G_BA.state_dict(), os.path.join(args.save_dir, f'G_BA_epoch{epoch+1}.pth'))
            torch.save(D_A.state_dict(),  os.path.join(args.save_dir, f'D_A_epoch{epoch+1}.pth'))
            torch.save(D_B.state_dict(),  os.path.join(args.save_dir, f'D_B_epoch{epoch+1}.pth'))
            print(f"  → Checkpoint saved for epoch {epoch+1}")

        # Always save latest weights for inference
        torch.save(G_AB.state_dict(), os.path.join(args.save_dir, 'G_AB.pth'))
        torch.save(G_BA.state_dict(), os.path.join(args.save_dir, 'G_BA.pth'))

    print("Training complete!")


if __name__ == "__main__":
    train()
