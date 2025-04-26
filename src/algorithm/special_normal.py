""" with values clipped at the ends to fit within high and low of -1.00 to 1.00"""

import torch
import torch.distributions as pyd
import matplotlib.pyplot as plt

# Helper: standard normal sample
def _standard_normal(shape, dtype, device):
    return torch.randn(shape, dtype=dtype, device=device)

# Your class
class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape,
                               dtype=self.loc.dtype,
                               device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)

# Instantiate and sample
loc = torch.tensor(0.0)
scale = torch.tensor(0.5)
dist = TruncatedNormal(loc, scale, low=-1.0, high=1.0)

samples = dist.sample(sample_shape=(10000,))  # draw 10,000 samples

# Plot
plt.hist(samples.numpy(), bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
plt.title("Truncated Normal Distribution")
plt.xlabel("Value")
plt.ylabel("Density")
plt.grid(True)
plt.show()


