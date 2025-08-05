import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import cv2
from scipy.ndimage.filters import gaussian_filter
import os
import sys
import time

# To handle a common Matplotlib issue on some systems
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#------------------------------------------------------------------
# Image dimensions
xs = 800
ys = 800

def mask(image, size):
  """Applies an unsharp mask to the image."""
  blur = gaussian_filter(image, size)
  return (image - blur * 0.83)

def show_array(array, name):
    """Displays an image using OpenCV."""
    bg = np.percentile(array, 0.1)
    array = array - bg
    array = array / np.percentile(array, 99.8)
    
    cv2.imshow(name, array*0.9)
    cv2.waitKey(1)

#------------------------------------------------------------------
# Cropping dimensions
X0 = 60
Y0 = 60
XS = 700
YS = 700

def benoit_noise(image):
  """
  Calculates a noise metric based on the difference between a pixel and its 8 neighbors.
  This acts as a regularization term to encourage smoothness.
  """
  dnoise = 0
  # Direct neighbors
  dnoise = dnoise + torch.mean(torch.square(image - torch.roll(image, shifts=(1, 0), dims=(0, 1))))
  dnoise = dnoise + torch.mean(torch.square(image - torch.roll(image, shifts=(0, 1), dims=(0, 1))))
  dnoise = dnoise + torch.mean(torch.square(image - torch.roll(image, shifts=(0, -1), dims=(0, 1))))
  dnoise = dnoise + torch.mean(torch.square(image - torch.roll(image, shifts=(-1, 0), dims=(0, 1))))
  # Diagonal neighbors
  dnoise = dnoise + 0.5 * torch.mean(torch.square(image - torch.roll(image, shifts=(1, 1), dims=(0, 1))))
  dnoise = dnoise + 0.5 * torch.mean(torch.square(image - torch.roll(image, shifts=(-1, 1), dims=(0, 1))))
  dnoise = dnoise + 0.5 * torch.mean(torch.square(image - torch.roll(image, shifts=(1, -1), dims=(0, 1))))
  dnoise = dnoise + 0.5 * torch.mean(torch.square(image - torch.roll(image, shifts=(-1, -1), dims=(0, 1))))

  return torch.sqrt(dnoise)

#------------------------------------------------------------------
# PSF kernel size
P = 21
P2 = 10

def get_image(fn, device):
    """Reads and preprocesses a FITS image."""
    with fits.open(fn) as hdul:
        hdul.verify('fix')
        x = hdul[0].data
    x = x[X0:X0+XS, Y0:Y0+YS]
    x = x.astype(np.float32)

    x = x / np.percentile(x, 90.0)
    x = x - np.percentile(x, 1.0)
    return torch.from_numpy(x).to(device)

#------------------------------------------------------------------

class DeconvolutionModel(nn.Module):
    def __init__(self, initial_model_numpy):
        super(DeconvolutionModel, self).__init__()
        self.model = nn.Parameter(torch.from_numpy(initial_model_numpy.copy()))
        
        k0 = create_kernel(0, 0)
        k1 = create_kernel(2, 0)
        k2 = create_kernel(0, -2)
        k3 = create_kernel(0, 0)

        self.psf0 = nn.Parameter(torch.from_numpy(k0))
        self.psf1 = nn.Parameter(torch.from_numpy(k1))
        self.psf2 = nn.Parameter(torch.from_numpy(k2))
        self.psf3 = nn.Parameter(torch.from_numpy(k3))
        
        self.mul0 = nn.Parameter(torch.tensor(1.0))
        self.mul1 = nn.Parameter(torch.tensor(1.0))
        self.mul2 = nn.Parameter(torch.tensor(1.0))
        self.mul3 = nn.Parameter(torch.tensor(1.0))

    def forward(self, observed0, observed1, observed2, observed3):
        model_unsqueezed = self.model.unsqueeze(0).unsqueeze(0)

        p0 = torch.abs(self.psf0).unsqueeze(0).unsqueeze(0)
        p1 = torch.abs(self.psf1).unsqueeze(0).unsqueeze(0)
        p2 = torch.abs(self.psf2).unsqueeze(0).unsqueeze(0)
        p3 = torch.abs(self.psf3).unsqueeze(0).unsqueeze(0)

        r0 = self.mul0 * F.conv2d(model_unsqueezed, p0, padding='valid').squeeze()
        r1 = self.mul1 * F.conv2d(model_unsqueezed, p1, padding='valid').squeeze()
        r2 = self.mul2 * F.conv2d(model_unsqueezed, p2, padding='valid').squeeze()
        r3 = self.mul3 * F.conv2d(model_unsqueezed, p3, padding='valid').squeeze()

        obs0_cropped = observed0[P2:XS-P2, P2:YS-P2]
        obs1_cropped = observed1[P2:XS-P2, P2:YS-P2]
        obs2_cropped = observed2[P2:XS-P2, P2:YS-P2]
        obs3_cropped = observed3[P2:XS-P2, P2:YS-P2]

        e0 = torch.mean(torch.square(r0 - obs0_cropped))
        e1 = torch.mean(torch.square(r1 - obs1_cropped))
        e2 = torch.mean(torch.square(r2 - obs2_cropped))
        e3 = torch.mean(torch.square(r3 - obs3_cropped))

        noise_reg = benoit_noise(self.model)
        noise_reg = torch.clamp(noise_reg - 0.086562345, 0, 100.0)

        loss = 100.0 * (e0 + e1 + e2 + e3) + 30.0 * noise_reg
        
        return loss, e0, e1, e2, e3, noise_reg

def create_kernel(offset_x, offset_y, sigma=2.1):
    k = np.zeros((P, P), dtype=np.float32)
    k[P2 + offset_y, P2 + offset_x] = 1.0
    k = gaussian_filter(k, sigma=sigma) + 0.0001
    k = k / np.sum(k)
    return k

#------------------------------------------------------------------

def optimize():
    np.set_printoptions(linewidth=285, nanstr='nan', precision=2, suppress=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load images and move to device
    im0_numpy = get_image.__globals__['get_image']("./i1.fits", device).cpu().numpy()
    im1_numpy = get_image.__globals__['get_image']("./i3.fits", device).cpu().numpy()
    im2_numpy = get_image.__globals__['get_image']("./i2.fits", device).cpu().numpy()
    im3_numpy = get_image.__globals__['get_image']("./i4.fits", device).cpu().numpy()
    
    initial_model_numpy = (im0_numpy + im1_numpy + im2_numpy + im3_numpy) / 4.0

    observed0 = torch.from_numpy(im0_numpy).to(device)
    observed1 = torch.from_numpy(im1_numpy).to(device)
    observed2 = torch.from_numpy(im2_numpy).to(device)
    observed3 = torch.from_numpy(im3_numpy).to(device)

    deconv_model = DeconvolutionModel(initial_model_numpy).to(device)
    
    optimizer0 = optim.Adam([deconv_model.model, deconv_model.psf0, deconv_model.mul0, deconv_model.mul1, deconv_model.mul2, deconv_model.mul3], lr=0.0001)
    optimizer1 = optim.Adam([deconv_model.model, deconv_model.psf1, deconv_model.mul0, deconv_model.mul1, deconv_model.mul2, deconv_model.mul3], lr=0.0001)
    optimizer2 = optim.Adam([deconv_model.model, deconv_model.psf2, deconv_model.mul0, deconv_model.mul1, deconv_model.mul2, deconv_model.mul3], lr=0.0001)
    optimizer3 = optim.Adam([deconv_model.model, deconv_model.psf3, deconv_model.mul0, deconv_model.mul1, deconv_model.mul2, deconv_model.mul3], lr=0.0001)

    for k in range(11100):
        for i in range(600):
            if 0 < i <= 150:
                optimizer = optimizer0
            elif 150 < i <= 300:
                optimizer = optimizer1
            elif 300 < i <= 450:
                optimizer = optimizer2
            elif 450 < i < 600:
                optimizer = optimizer3
            else:
                continue

            optimizer.zero_grad()
            loss, e0, e1, e2, e3, noise_reg = deconv_model(observed0, observed1, observed2, observed3)
            loss.backward()
            optimizer.step()

            print(f"\rEpoch {k}/{11100}, Iteration {i}/{600}, Loss: {loss.item():.4f}", end="")
            sys.stdout.flush()

            if i % 150 == 0:
                print(f"\n{k}, {i}, loss: {loss.item():.4f}, "
                      f"e0: {e0.item():.4f}, e1: {e1.item():.4f}, "
                      f"e2: {e2.item():.4f}, e3: {e3.item():.4f}, "
                      f"mul0: {deconv_model.mul0.item():.2f}, mul1: {deconv_model.mul1.item():.2f}, "
                      f"noise: {noise_reg.item():.4f}")

                model_img = deconv_model.model.detach().cpu().numpy()
                model_img = model_img[20:-20, 20:-20]
                model_img = model_img - np.min(model_img)
                model_img = model_img / np.percentile(model_img, 90)
                show_array(mask(model_img, 3), "model")

                psf0_img = deconv_model.psf0.detach().cpu().numpy()
                psf0_img /= np.max(psf0_img)
                cv2.imshow("psf0", psf0_img * 0.8)

                psf1_img = deconv_model.psf1.detach().cpu().numpy()
                psf1_img /= np.max(psf1_img)
                cv2.imshow("psf1", psf1_img * 0.8)

                psf2_img = deconv_model.psf2.detach().cpu().numpy()
                psf2_img /= np.max(psf2_img)
                cv2.imshow("psf2", psf2_img * 0.8)

                psf3_img = deconv_model.psf3.detach().cpu().numpy()
                psf3_img /= np.max(psf3_img)
                cv2.imshow("psf3", psf3_img * 0.8)
                
                with torch.no_grad():
                    model_unsqueezed = deconv_model.model.unsqueeze(0).unsqueeze(0)
                    p2_unsqueezed = torch.abs(deconv_model.psf2).unsqueeze(0).unsqueeze(0)
                    r2 = deconv_model.mul2 * F.conv2d(model_unsqueezed, p2_unsqueezed, padding='valid').squeeze()
                    obs2_cropped = observed2[P2:XS-P2, P2:YS-P2]
                    v2 = torch.square(r2 - obs2_cropped)
                    v2_img = v2.cpu().numpy()
                    v2_img /= np.max(v2_img)
                    cv2.imshow("val2", v2_img)

                cv2.waitKey(1)

#------------------------------------------------------------------

if __name__ == '__main__':
    optimize()
