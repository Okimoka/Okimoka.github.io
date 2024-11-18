---
layout: post
title:  "Timewarp Blogpost"
date:   2024-11-18 15:21:05 +0100
permalink: /timewarp
---
## Introduction
- Quick Overview of Molecular Dynamics
	- Importance of MD
	- Challenges of standard MD simulation
		- Impossibly slow simulation for larger timespans (necessary to observe e.g. folding)
		- No transferability between different molecular systems
	- Underlying physics
		- Significance of the Boltzmann distribution for MD
		- Langevin dynamics
	- Example
		- Possible desired input/output pair to motivate the problem
			- Introduce notation
		- Intuition for methodology used to tackle the problem
			- Especially regarding the goal of transferability

## Foundations

- MCMC + Metropolis Hastings + Gibbs sampling
	- Explaination based on the Probabilistic ML lecture (explain pseudocode)
- Transformers
	- Explaination based on the Foundation Models lecture
	- Specially go into multihead kernel self-attention (needed later)
- (Conditional) Normalizing Flows
	- Explaination based on https://arxiv.org/pdf/1505.05770
	- Connection to MCMC
	- RealNVP
		-Explaination based on https://lilianweng.github.io/posts/2018-10-13-flow-models/

## Putting everything together

![Full architecture](assets/architecture.png)

- MCMC + MH
	- Used proposal distribution
	- Explaination of the full adapted algorithm
		- Batch sampling
- Normalizing Flow
	- Used base distribution
	- Used diffeomorphisms
		- Discarding of the MD velocity
	- Used Dataset
	- Used transformations for coupling layers om RealNVP
	- Satisfaction of physical symmetries
- Transformer
	- Used input vectors and structure
	- Adaptations made
		- Kernel self-attention
		- Others
- Training (Loss functions)
	- Likelihood training
	- Acceptance training

## Conclusion
