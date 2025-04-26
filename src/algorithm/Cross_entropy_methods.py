import numpy as np
import matplotlib.pyplot as plt


# Target function: simple 2D quadratic
def objective_function(x):
    return -np.sum((x - 2) ** 2, axis=-1)  # maximum at (2, 2)


# Cross Entropy Method
def cem(f, n_iterations, batch_size, elite_frac, initial_mean, initial_std, n_dim):
    mean = np.copy(initial_mean)
    std = np.copy(initial_std)
    history = []

    n_elite = int(np.round(batch_size * elite_frac))

    for iteration in range(n_iterations):
        samples = np.random.randn(batch_size, n_dim) * std + mean
        rewards = f(samples)

        elite_idxs = rewards.argsort()[-n_elite:]  # best rewards
        elite_samples = samples[elite_idxs]

        mean = elite_samples.mean(axis=0)
        std = elite_samples.std(axis=0)

        history.append((mean.copy(), rewards.max()))

        if iteration % 5 == 0:
            print(f"Iteration {iteration}: Best reward {rewards.max():.2f}")

    return mean, history


# Settings
n_dim = 2
n_iterations = 50
batch_size = 100
elite_frac = 0.2
initial_mean = np.zeros(n_dim)
initial_std = np.ones(n_dim)

# Run CEM
final_mean, history = cem(objective_function, n_iterations, batch_size, elite_frac, initial_mean, initial_std, n_dim)

# Extract history
means, best_rewards = zip(*history)

# Plotting
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# 1. Plot function landscape
x = np.linspace(-2, 5, 100)
y = np.linspace(-2, 5, 100)
X, Y = np.meshgrid(x, y)
Z = objective_function(np.stack([X, Y], axis=-1))

axs[0].contourf(X, Y, Z, levels=50, cmap='viridis')
axs[0].set_title('Objective Function Landscape')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')

# 2. Plot optimization trajectory
means_array = np.array(means)
axs[0].plot(means_array[:, 0], means_array[:, 1], marker='o', color='red')

# 3. Mean reward over iterations
axs[1].plot(best_rewards)
axs[1].set_title('Best Reward Over Iterations')
axs[1].set_xlabel('Iteration')
axs[1].set_ylabel('Best Reward')

# 4. Plot samples in final iteration
samples = np.random.randn(batch_size, n_dim) * initial_std + final_mean
axs[2].contourf(X, Y, Z, levels=50, cmap='viridis')
axs[2].scatter(samples[:, 0], samples[:, 1], color='white', alpha=0.7, label='Samples')
axs[2].scatter(final_mean[0], final_mean[1], color='red', s=100, label='Final Mean')
axs[2].legend()
axs[2].set_title('Samples at Final Iteration')
axs[2].set_xlabel('x')
axs[2].set_ylabel('y')

plt.tight_layout()
plt.show()
