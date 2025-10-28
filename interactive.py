import torch
import torch.nn.functional as F
import mrcfile
import numpy as np
from scipy.spatial.transform import Rotation as R

# --- Device ---
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Utility Functions ---
def euler2rot_batch(angles):
    """Convert batch of Euler angles (ZYZ) to rotation matrices."""
    N = angles.shape[0]
    mats = []
    for i in range(N):
        r = R.from_euler('zyz', angles[i].cpu().numpy(), degrees=True)
        mats.append(torch.tensor(r.as_matrix(), device=device, dtype=torch.float32))
    return torch.stack(mats, dim=0)  # [N,3,3]

def project_volume_batch(volume, rot_mats):
    """Project 3D volume for a batch of rotation matrices. volume: [D,H,W], rot_mats: [N,3,3]"""
    D,H,W = volume.shape
    N = rot_mats.shape[0]
    grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    coords = torch.stack([grid_x, grid_y, torch.zeros_like(grid_x)], dim=-1).float()
    coords_centered = coords - torch.tensor([W/2,H/2,D/2], device=device)
    coords_flat = coords_centered.view(-1,3)
    proj_batch = []
    for i in range(N):
        coords_rot = coords_flat @ rot_mats[i].T + torch.tensor([W/2,H/2,D/2], device=device)
        x = 2*coords_rot[:,0]/(W-1) - 1
        y = 2*coords_rot[:,1]/(H-1) - 1
        grid = torch.stack([x,y], dim=-1).view(1,H,W,2)
        vol_4d = volume.unsqueeze(0).unsqueeze(0)
        proj = F.grid_sample(vol_4d, grid, mode='bilinear', padding_mode='border', align_corners=True)
        proj_batch.append(proj[0,0])
    return torch.stack(proj_batch, dim=0)  # [N,H,W]

def apply_anisotropic_magnification_batch(images, M):
    """Apply 2x2 magnification matrix to batch of images."""
    N,H,W = images.shape
    y, x = torch.meshgrid(torch.linspace(-1,1,H, device=device), torch.linspace(-1,1,W, device=device), indexing='ij')
    grid = torch.stack([x.flatten(), y.flatten()], dim=1).T
    coords = (grid * torch.tensor([[W/2],[H/2]], device=device))
    coords = torch.linalg.inv(M) @ coords
    coords = coords / torch.tensor([[W/2],[H/2]], device=device)
    grid_sample = torch.stack([coords[0], coords[1]], dim=-1).T.view(H,W,2).unsqueeze(0).repeat(N,1,1,1)
    images_ = images.unsqueeze(1)
    sampled = F.grid_sample(images_, grid_sample, mode='bilinear', padding_mode='border', align_corners=True)
    return sampled[:,0]

def fourier_cc_loss_batched(M_flat, particle_images, align3d, volume, batch_size=32):
    """
    Batched Fourier-space CC loss for very large datasets.
    Processes particles in smaller batches to fit GPU memory.
    """
    M = M_flat.view(2,2)
    N_total = particle_images.shape[0]
    loss_total = 0.0
    for start in range(0, N_total, batch_size):
        end = min(start + batch_size, N_total)
        im_batch = particle_images[start:end]
        align_batch = align3d[start:end]
        rot_mats = euler2rot_batch(align_batch[:,:3])
        proj_batch = project_volume_batch(volume, rot_mats)
        proj_batch = apply_anisotropic_magnification_batch(proj_batch, M)
        
        shifts = torch.tensor(align_batch[:,3:5], device=device)
        F_proj = torch.fft.fft2(proj_batch)
        F_im = torch.fft.fft2(im_batch)
        H,W = im_batch.shape[1], im_batch.shape[2]
        ky = torch.fft.fftfreq(H,1.0).to(device)
        kx = torch.fft.fftfreq(W,1.0).to(device)
        KX,KY = torch.meshgrid(kx, ky, indexing='ij')
        phase = torch.exp(-2j*np.pi*(KX.unsqueeze(0)*shifts[:,0].unsqueeze(1).unsqueeze(2) +
                                     KY.unsqueeze(0)*shifts[:,1].unsqueeze(1).unsqueeze(2)))
        F_proj_shifted = F_proj * phase
        cc = torch.real(torch.fft.ifft2(F_proj_shifted * torch.conj(F_im))).sum(dim=(1,2))
        loss_total += -cc.sum()
    return loss_total

# --- Load Inputs ---
volume_file = "volume.mrc" #placeholders
align_file = "align3d.npy" #placeholders
particle_files = ["particle_0001.mrc","particle_0002.mrc"]  # extend to all particles #placeholders

with mrcfile.open(volume_file) as f:
    volume = torch.tensor(f.data.astype(np.float32), device=device)

align3d = np.load(align_file)
particle_images = [torch.tensor(mrcfile.open(f).data.astype(np.float32), device=device) for f in particle_files]
particle_images = torch.stack(particle_images, dim=0)

# --- First Coarse Optimization ---
M0 = torch.eye(2, device=device).flatten().requires_grad_(True)
optimizer = torch.optim.LBFGS([M0], max_iter=50, line_search_fn='strong_wolfe')

def closure():
    optimizer.zero_grad()
    loss = fourier_cc_loss_batched(M0, particle_images, align3d, volume, batch_size=16)
    loss.backward()
    return loss

optimizer.step(closure)
best_M = M0.detach().view(2,2)
print("Coarse optimization result:\n", best_M.cpu().numpy())

# --- Second Fine Optimization ---
M0 = best_M.clone().detach().requires_grad_(True)
optimizer = torch.optim.LBFGS([M0], max_iter=30, line_search_fn='strong_wolfe')
optimizer.step(closure)
best_M_refined = M0.detach().view(2,2)
print("Refined anisotropic magnification matrix:\n", best_M_refined.cpu().numpy())
