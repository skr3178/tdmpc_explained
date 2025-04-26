import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal


__REDUCE__ = lambda b: 'mean' if b else 'none'


def l1(pred, target, reduce=False):
	"""Computes the L1-loss between predictions and targets."""
	# Priority loss updates
	return F.l1_loss(pred, target, reduction=__REDUCE__(reduce))


def mse(pred, target, reduce=False):
	"""Computes the MSE loss between predictions and targets."""
	# used to compute reward loss, consistency loss and value loss x 2
	return F.mse_loss(pred, target, reduction=__REDUCE__(reduce))


def _get_out_shape(in_shape, layers):
	"""Utility function. Returns the output shape of a network for a given input shape."""
	# used in encoder, outcome of CNN layer
	x = torch.randn(*in_shape).unsqueeze(0)
	return (nn.Sequential(*layers) if isinstance(layers, list) else layers)(x).squeeze(0).shape


def orthogonal_init(m):
	"""Orthogonal layer initialization.- better for stability and adding some values for intial run of NN"""
	# h-NN initialization
	if isinstance(m, nn.Linear):
		nn.init.orthogonal_(m.weight.data)
		if m.bias is not None:
			nn.init.zeros_(m.bias)
	elif isinstance(m, nn.Conv2d):
		gain = nn.init.calculate_gain('relu')
		nn.init.orthogonal_(m.weight.data, gain)
		if m.bias is not None:
			nn.init.zeros_(m.bias)


def ema(m, m_target, tau):
	"""Update slow-moving average of online network (target network) at rate tau."""
	# Copy weight directly from RL to online can cause stability
	# used while updating the h network/term, from online to regular
	with torch.no_grad():
		for p, p_target in zip(m.parameters(), m_target.parameters()):
			#Copy weight and biases from m NN to m_target NN"""
			p_target.data.lerp_(p.data, tau)
			#Linear_interpolation_function: p_target = (1 - tau) * p_target + tau * p"""


def set_requires_grad(net, value):
	"""Enable/disable gradients for a given (sub)network."""
	# used in track q grad- as NO
	for param in net.parameters():
		param.requires_grad_(value)


class TruncatedNormal(pyd.Normal):
	"""Utility class implementing the truncated normal distribution."""
	# used in TOLD function, pi
	def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
		super().__init__(loc, scale, validate_args=False)
		self.low = low
		self.high = high
		self.eps = eps

	def _clamp(self, x):
		"""clamps all elements of x within a certain range of low+epsilon(-1+1e-6), high-epsilon(1-1e-6)"""
		#clamped_x is a tensor"""
		clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
		#x.detach() Does Returns a new tensor with the same data as x but no gradient connection to its history."""
		#standard clamping kills gradients at the boundaries"""
		x = x - x.detach() + clamped_x.detach()
		return x

	def sample(self, clip=None, sample_shape=torch.Size()):
		"""Determines the desired shape of sample, e.g 100 for 100 samples"""
		shape = self._extended_shape(sample_shape)
		#samples from standard normal distribution with mean 0 and standard deviation 1."""
		eps = _standard_normal(shape,
							   dtype=self.loc.dtype,
							   device=self.loc.device)
		#adding noise and thereafter clipping it"""
		eps *= self.scale
		if clip is not None:
			eps = torch.clamp(eps, -clip, clip)
		x = self.loc + eps
		return self._clamp(x)


class NormalizeImg(nn.Module):
	"""Normalizes pixel observations to [0,1) range."""
	# used in encoder input normalisation
	def __init__(self):
		super().__init__()

	def forward(self, x):
		return x.div(255.)


class Flatten(nn.Module):
	"""Flattens its input to a (batched) vector."""
	def __init__(self):
		super().__init__()
		
	def forward(self, x):
		return x.view(x.size(0), -1)


def enc(cfg):
	"""Returns a TOLD encoder."""
	# Encoder is used for the purpose of extracting the most essential features of pixel observations.
	# This is converted to latent observations-flattened to a single layer later. In case it is not pixels, other functions follow
	if cfg.modality == 'pixels':
		C = int(3*cfg.frame_stack)
		layers = [NormalizeImg(),
				  nn.Conv2d(C, cfg.num_channels, 7, stride=2), nn.ReLU(),
				  nn.Conv2d(cfg.num_channels, cfg.num_channels, 5, stride=2), nn.ReLU(),
				  nn.Conv2d(cfg.num_channels, cfg.num_channels, 3, stride=2), nn.ReLU(),
				  nn.Conv2d(cfg.num_channels, cfg.num_channels, 3, stride=2), nn.ReLU()]
		out_shape = _get_out_shape((C, cfg.img_size, cfg.img_size), layers)
		layers.extend([Flatten(), nn.Linear(np.prod(out_shape), cfg.latent_dim)])
	else:
		# when modality is not pixel, for e.g. for quadruped-run, modality ='state', MLP is used
		# cfg.obs_shape[0]: the size of the input observation vector (e.g., 17-dimensional state like joint positions, velocities, etc.
		# 2-layer MLP	Sufficient for structured, low-dim input
		layers = [nn.Linear(cfg.obs_shape[0], cfg.enc_dim), nn.ELU(),
				  nn.Linear(cfg.enc_dim, cfg.latent_dim)]
	return nn.Sequential(*layers)


def mlp(in_dim, mlp_dim, out_dim, act_fn=nn.ELU()):
	"""Returns an MLP, A general purpose neural network."""
	# used in defining the dynamics, rewards and pi functions in TD-MPC
	if isinstance(mlp_dim, int):
		#If input is an integer[64], creates a 2x hidden layers of same size, else if already a list,e.g [128, 64] takes as is."""
		mlp_dim = [mlp_dim, mlp_dim]
		# follows typical config {linear-activation, lin- act, lin_output}
	return nn.Sequential(
		nn.Linear(in_dim, mlp_dim[0]), act_fn,
		nn.Linear(mlp_dim[0], mlp_dim[1]), act_fn,
		nn.Linear(mlp_dim[1], out_dim))


def q(cfg, act_fn=nn.ELU()):
	"""Returns a Q-function that uses Layer Normalization."""
	# In reinforcement learning, the Q-function (a.k.a. action-value function) estimates: Q(s,a)
	# Q(s,a)- expected return from taking action a in state s
	# Outputs a scalar → the estimated Q-value of the (state, action) pair.
	# Input: [latent_dim + action_dim] concatenated
	# → Linear → LayerNorm → Tanh
	# → Linear → ELU
	# → Linear → 1(Q - value)
	return nn.Sequential(nn.Linear(cfg.latent_dim+cfg.action_dim, cfg.mlp_dim),
						 #  LayerNorm Stabilizes activations across features, prevents exploding-vanishing grads
						 # Layer normalization is very effective at stabilizing the hidden state dynamics in recurrent networks.
						 nn.LayerNorm(cfg.mlp_dim), nn.Tanh(), #squash to [-1, 1]-centered around 0, makes convergence and training faster
						 nn.Linear(cfg.mlp_dim, cfg.mlp_dim), nn.ELU(),
						 nn.Linear(cfg.mlp_dim, 1))


class RandomShiftsAug(nn.Module):
	"""
	Random shift image augmentation.
	Adapted from https://github.com/facebookresearch/drqv2
	"""

	# Random Shift Augmentation: Randomly shifts the input image by a few pixels (with padding) to simulate small perturbations.
	# This helps the model generalize better by seeing slightly varied versions of the same observation.
	# Similar to Image training where introduction of rotation, pixel density reduction create a more robust trained model
	def __init__(self, cfg):
		super().__init__()
		self.pad = int(cfg.img_size/21) if cfg.modality == 'pixels' else None

	def forward(self, x):
		if not self.pad:
			return x
		n, c, h, w = x.size()
		assert h == w
		padding = tuple([self.pad] * 4)
		x = F.pad(x, padding, 'replicate')
		eps = 1.0 / (h + 2 * self.pad)
		arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
		arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
		base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
		base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
		shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
		shift *= 2.0 / (h + 2 * self.pad)
		grid = base_grid + shift
		return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)


class Episode(object):
	"""Storage object for a single episode."""
	def __init__(self, cfg, init_obs):
		self.cfg = cfg
		self.device = torch.device(cfg.device)
		dtype = torch.float32 if cfg.modality == 'state' else torch.uint8
		# +1 because the first observation comes before the first action.
		self.obs = torch.empty((cfg.episode_length+1, *init_obs.shape), dtype=dtype, device=self.device)
		self.obs[0] = torch.tensor(init_obs, dtype=dtype, device=self.device)
		self.action = torch.empty((cfg.episode_length, cfg.action_dim), dtype=torch.float32, device=self.device)
		self.reward = torch.empty((cfg.episode_length,), dtype=torch.float32, device=self.device)
		self.cumulative_reward = 0
		self.done = False
		self._idx = 0
	
	def __len__(self):
		# length defined by how many steps have been added
		return self._idx

	@property
	def first(self):
		# if len == 0, then first is true
		return len(self) == 0
	
	def __add__(self, transition):
		self.add(*transition)
		return self

	def add(self, obs, action, reward, done):
		# adds obs[i] to original obs space
		# adds action to original action space
		# adds reward to original reward space
		self.obs[self._idx+1] = torch.tensor(obs, dtype=self.obs.dtype, device=self.obs.device)
		self.action[self._idx] = action
		self.reward[self._idx] = reward
		self.cumulative_reward += reward
		# when does done turn into "True"? perhaps later in the algo, there is a program that computes...
		# ...obs, reward, done, info = env.step(action)
		self.done = done
		self._idx += 1


class ReplayBuffer():
	"""
	Storage and sampling functionality for training TD-MPC / TOLD.
	The replay buffer is stored in GPU memory when training from state.
	Uses prioritized experience replay by default."""
	def __init__(self, cfg):
		# intializes Observation storage (state or image-based), Action storage, Reward storage, Priority storage,
		# Tracking indices and buffer fullness
		self.cfg = cfg
		self.device = torch.device(cfg.device)
		self.capacity = min(cfg.train_steps, cfg.max_buffer_size)
		dtype = torch.float32 if cfg.modality == 'state' else torch.uint8
		#Takes all observations in the episode except the last one ([:-1])
		#The last observation (s_T) is only used as the terminal "next state" for the final transition
		obs_shape = cfg.obs_shape if cfg.modality == 'state' else (3, *cfg.obs_shape[-2:])
		self._obs = torch.empty((self.capacity+1, *obs_shape), dtype=dtype, device=self.device)
		self._last_obs = torch.empty((self.capacity//cfg.episode_length, *cfg.obs_shape), dtype=dtype, device=self.device)
		self._action = torch.empty((self.capacity, cfg.action_dim), dtype=torch.float32, device=self.device)
		self._reward = torch.empty((self.capacity,), dtype=torch.float32, device=self.device)
		# Priorities are used for Prioritized Experience Replay (PER) —
		# a method to sample more important transitions more frequently during training.
		self._priorities = torch.ones((self.capacity,), dtype=torch.float32, device=self.device)
		self._eps = 1e-6
		self._full = False
		self.idx = 0

	def __add__(self, episode: Episode):
		#This method allows you to use the + operator to add an Episode to the replay buffer.
		#instead of buffer.add(episode) we can use buffer = buffer + episode
		self.add(episode)
		return self

	def add(self, episode: Episode):
		self._obs[self.idx:self.idx+self.cfg.episode_length] = episode.obs[:-1] if self.cfg.modality == 'state' else episode.obs[:-1, -3:]
		self._last_obs[self.idx//self.cfg.episode_length] = episode.obs[-1]
		self._action[self.idx:self.idx+self.cfg.episode_length] = episode.action
		self._reward[self.idx:self.idx+self.cfg.episode_length] = episode.reward
		if self._full:
			max_priority = self._priorities.max().to(self.device).item()
		else:
			#New transitions are assigned the current maximum priority in the buffer (or 1.0 if empty).
			max_priority = 1. if self.idx == 0 else self._priorities[:self.idx].max().to(self.device).item()
		mask = torch.arange(self.cfg.episode_length) >= self.cfg.episode_length-self.cfg.horizon
		new_priorities = torch.full((self.cfg.episode_length,), max_priority, device=self.device)
		# zero priorities to horizon ensure that the near the end transitions have lower priority. Updated once the TD error is known.
		# Excluded from sampling. Update priorities later updates these samples. Transitions near episode termination have lower priority
		new_priorities[mask] = 0 ## Zero-priority for last 'horizon' steps
		self._priorities[self.idx:self.idx+self.cfg.episode_length] = new_priorities
		self.idx = (self.idx + self.cfg.episode_length) % self.capacity
		self._full = self._full or self.idx == 0

	def update_priorities(self, idxs, priorities):
		self._priorities[idxs] = priorities.squeeze(1).to(self.device) + self._eps # eps prevents zero priority,
		# ensures no transition is permanently excluded.

	def _get_obs(self, arr, idxs):
		if self.cfg.modality == 'state':
			return arr[idxs]
		obs = torch.empty((self.cfg.batch_size, 3*self.cfg.frame_stack, *arr.shape[-2:]), dtype=arr.dtype, device=torch.device('cuda'))
		obs[:, -3:] = arr[idxs].cuda()
		_idxs = idxs.clone()
		mask = torch.ones_like(_idxs, dtype=torch.bool)
		for i in range(1, self.cfg.frame_stack):
			mask[_idxs % self.cfg.episode_length == 0] = False
			_idxs[mask] -= 1
			obs[:, -(i+1)*3:-i*3] = arr[_idxs].cuda()
		return obs.float()

	def sample(self):
		#Priorities are used to bias the sampling of transitions toward those that are expected to contribute more to learning
		# (typically, those with larger TD errors or higher surprise).
		# priority is per transition not episode
		probs = (self._priorities if self._full else self._priorities[:self.idx]) ** self.cfg.per_alpha
		# Alpha controls prioritization strength (0=uniform, 1=full)
		probs /= probs.sum() # Convert to probability distribution
		total = len(probs)
		#(corrects bias)
		idxs = torch.from_numpy(np.random.choice(total, self.cfg.batch_size, p=probs.cpu().numpy(), replace=not self._full)).to(self.device)
		weights = (total * probs[idxs]) ** (-self.cfg.per_beta)
		weights /= weights.max()

		obs = self._get_obs(self._obs, idxs)
		next_obs_shape = self._last_obs.shape[1:] if self.cfg.modality == 'state' else (3*self.cfg.frame_stack, *self._last_obs.shape[-2:])
		next_obs = torch.empty((self.cfg.horizon+1, self.cfg.batch_size, *next_obs_shape), dtype=obs.dtype, device=obs.device)
		action = torch.empty((self.cfg.horizon+1, self.cfg.batch_size, *self._action.shape[1:]), dtype=torch.float32, device=self.device)
		reward = torch.empty((self.cfg.horizon+1, self.cfg.batch_size), dtype=torch.float32, device=self.device)
		for t in range(self.cfg.horizon+1):
			_idxs = idxs + t
			next_obs[t] = self._get_obs(self._obs, _idxs+1)
			action[t] = self._action[_idxs]
			reward[t] = self._reward[_idxs]

		mask = (_idxs+1) % self.cfg.episode_length == 0
		next_obs[-1, mask] = self._last_obs[_idxs[mask]//self.cfg.episode_length].cuda().float()
		if not action.is_cuda:
			action, reward, idxs, weights = \
				action.cuda(), reward.cuda(), idxs.cuda(), weights.cuda()

		return obs, next_obs, action, reward.unsqueeze(2), idxs, weights


def linear_schedule(schdl, step):
	"""
	Outputs values following a linear decay schedule.
	Adapted from https://github.com/facebookresearch/drqv2
	"""
	try:
		return float(schdl)
	except ValueError:
		match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
		if match:
			init, final, duration = [float(g) for g in match.groups()]
			mix = np.clip(step / duration, 0.0, 1.0)
			return (1.0 - mix) * init + mix * final
	raise NotImplementedError(schdl)

# works in this way std_schedule: linear(0.5, ${min_std}, 25000), min_std: 0.05
#This is often used to reduce exploration noise (std) as the agent gets better at the task.
# This means:Start at 0.5 std, Linearly decay to 0.05, Over 25000 steps