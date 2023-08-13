import torch
import numpy as np
import matplotlib.pyplot as plt

print ("PyTorch Version:", torch.__version__)

# Device Config
device = torch.device('mps')

# Grid for computing image, subdivide the space
X, Y = np.mgrid[-4.0:4:0.01, -4.0:4:0.01]

# Loading in PyTorch tensors
x = torch.Tensor(X)
y = torch.Tensor(Y)

# Transferring Numpy array to the GPU
x = x.to(device)
y = y.to(device)

r = x**2 + y**2

# aka standard deviation
sigma = 2

# Computing Gaussian Function
z_gaus = torch.exp(-r**2/sigma)

plt.imshow(z_gaus.cpu().numpy())

plt.tight_layout()
plt.show()

# Computing Sine Function
z_sin = torch.sin(x + y)

plt.imshow(z_sin.cpu().numpy())

plt.tight_layout()
plt.show()

# Computing Co-Sine Function
z_cos = torch.cos(x + y)

plt.imshow(z_cos.cpu().numpy())

plt.tight_layout()
plt.show()

# Multiplying the Sine, Co-Sine and Gaus Functions together
# to get approximation of Gabor Filter
z_gaus_sin = z_gaus * z_sin * z_cos


plt.imshow(z_gaus_sin.cpu().numpy())

plt.tight_layout()
plt.show()

# Similar output would occur with co-sine function expect
# different phase
