import torch
import matplotlib.pyplot as plt
class GradientPreservingClamp:
    def __init__(self, low=-1.0, high=1.0, eps=1e-6):
        self.low = low
        self.high = high
        self.eps = eps
    def _clamp(self, x):
        """Gradient-preserving clamp"""
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        return x - x.detach() + clamped_x.detach()
    def naive_clamp(self, x):
        """Standard clamp that kills gradients at boundaries"""
        return torch.clamp(x, self.low + self.eps, self.high - self.eps)
# Test the clamping
clamp = GradientPreservingClamp()
# Create test values spanning outside the bounds
test_values = torch.linspace(-2, 2, 100, requires_grad=True)
# Apply both clamping methods
naive_output = clamp.naive_clamp(test_values)
smart_output = clamp._clamp(test_values)
# Compute gradients (simulate loss.backward())
naive_output.sum().backward()
naive_grad = test_values.grad.clone()
test_values.grad.zero_()
smart_output.sum().backward()
smart_grad = test_values.grad.clone()
# Visualize
plt.figure(figsize=(12, 5))
# Plot outputs
plt.subplot(1, 2, 1)
plt.plot(test_values.detach().numpy(), naive_output.detach().numpy(), label='Naive Clamp')
plt.plot(test_values.detach().numpy(), smart_output.detach().numpy(), '--', label='Gradient-Preserving Clamp')
plt.title('Forward Pass Output')
plt.xlabel('Input Value')
plt.ylabel('Clamped Output')
plt.legend()
plt.grid(True)
# Plot gradients
plt.subplot(1, 2, 2)
plt.plot(test_values.detach().numpy(), naive_grad.numpy(), label='Naive Clamp Gradients')
plt.plot(test_values.detach().numpy(), smart_grad.numpy(), '--', label='Smart Clamp Gradients')
plt.title('Backward Pass Gradients')
plt.xlabel('Input Value')
plt.ylabel('Gradient')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()