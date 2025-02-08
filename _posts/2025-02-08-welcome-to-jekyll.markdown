---
layout: post
title:  "Timewarp Blogpost"
date:   2025-02-08 23:38:05 +0100
permalink: /timewarp
use_math: true
---

Being able to make accurate, fast predictions about the properties of molecules - without requiring absurd amounts of computing power - has to be one of the most exciting (and apt!) uses of deep learning today.
It allows us to understand the very foundation of our world, and enables better research in developing new drugs, new materials, or even exploring outer space.

I would like to introduce the paper "Timewarp: Transferable Acceleration of Molecular Dynamics by Learning Time-Coarsened Dynamics" published in [NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a598c367280f9054434fdcc227ce4d38-Abstract-Conference.html), which I find to be a very interesting paper making big strides in this area of science.

# The Problem

Say we have some molecule in time at some position p and velocity v: $x(t) = (x^p(t), x^v(t))$ and we want to gain insights into its folding behavior. For this, we might be interested in the state of the molecule after time $\tau$ has passed ($x(t+\tau)$).

> By the way, the shape of the folded molecule is the main factor in determining its function, behavior and processes, making it a huge area of interest especially for proteins

We have formulas from physics that allow us to exactly describe the behavior of atoms in space. In traditional molecular dynamics (MD) simulations, we directly apply these equations to $x(t)$, until we receive the molecule in the desired new state $x(t+\tau)$. Here, $\tau$ might be anywhere from a few femtoseconds to even milliseconds.

As you might imagine, this process is extremely slow. A naïve implementation is even completely infeasible when faced with certain molecules. What's more, for every new type of molecular system, we need different kinds of molecular simulations, which further complicates this process.

# The Solution

One formula of particular interest for us here is the **Boltzmann distribution**. It gives us the probability of the molecule $x(t)$ to be in a specific state given our molecular system. The system is characterized by its Temperature $T$, potential energy function $U$, kinetic energy function $K$, and Boltzmann's constant $k_B$.

$\mu(x^{p},x^{v})\propto\exp\left(-\frac{1}{k_{B}T}(U(x^{p})+K(x^{v}))\right),\quad\mu(x^{p})=\int\mu(x^{p},x^{v})\,\mathrm{d}x^{v}$

> The Boltzmann distribution was originally formulated by Ludwig Boltzmann while studying the statistical mechanics of gases in thermal equilibrium

To get an intuition for this formula, you might recall from school physics that molecules prefer to be in a state of low free energy. 
The formula closesly expresses this fact by assigning higher probabilities to states with lower free energy.

Consider that a folded molecule state corresponds to a local energy minimum, and you can see why this formula is of such central interest to us! Being able to efficiently sample from the Boltzmann distribution already turns out to be enough for learning about many important properties of molecules.

In fact, from now on we will only focus on how to efficiently sample from this distribution (which, given the complex energy landscape of a molecule, cannot usually be efficiently sampled from).

## Sampling from a distribution

Being able to accurately sample from a difficult distributions in a very fundamental problem in many areas of research focusing on probability distributions. One algorithm which has been used for over 80 years at this point, is **Markov Chain Monte Carlo**. This will also serve as the core algorithm used in this paper to sample from the Boltzmann distribution. To be more precise, the specific variant we will be using is called the **Metropolis–Hastings algorithm** (MH).

Before we perform MH, we need two further distributions:

- A proposal distribution $g(x)$. For now you can simply imagine a random walk (meaning given some state, output a random state in its vicinity)

- An acceptance distribution $A(x \rightarrow x')$ (sometimes written $A(x \rightarrow x')$ to signify the change from $x$ to $x'$), which you can picture as an "educated guesser" who tells you whether some state is good or no good

Now we can run the algorithm! Starting at timestep $t_0$ and some starting state $\theta_0$...



![MCMC Algorithm](assets/MCMC.png)

[Image source](https://www.researchgate.net/figure/llustration-of-Markov-Chain-Monte-Carlo-method_fig1_334001505)

...we use our proposal distribution to suggest a new state $\theta_1$. Again, this can be some random state. Our acceptance distribution then decides whether we accept or reject this new state (or with what probability we want to accept it). Depending on its decision, we update (or not) our current state $\theta$. From here, we simply repeat the whole process until our desired accuracy is reached.

Our result will use the "path" that $\theta$ has taken to construct a probability distribution, as you would do with a histogram. The more frequently we chose a $\theta$ that was in a specific bin, the higher the probability in our final distribution.

This is the whole algorithm! Some things you might gather from this:
- The acceptance distribution is the part that decides what kind of distribution will actually be generated by the whole process
- A good proposal distribution will lead to better states being generated, and thus more states being accepted, which consequently results in a much faster convergence

So all we need to do at this point is to find a good proposal distribution $g(x)$ and fitting acceptance distribution $A(x, x')$ to give us quick and correct results.

### The acceptance distribution $A(x, x')$

Luckily for us, the choice of $A(x, x')$ is rather trivial. So far I've omitted most of the math that goes into guaranteeing the algorithm actually gives us the right solution so I will keep it short. An acceptance distribution has to necessarily fulfil the following equation:

$\frac{A(x^{\prime},x)}{A(x,x^{\prime})}=\frac{P(x^{\prime})}{P(x)}\frac{g(x\mid x^{\prime})}{g(x^{\prime}\mid x)}$

This is in order to fulfil the _detailed balance_ property (i.e. it should be equally likely to transition from $x$ to $x'$ as it is to transition from $x'$ to $x$), as well as the uniqueness property of the stationary distribution. One choice for $A(x',x)$ which fulfils the equation is:

$A(x^{\prime},x)=\mathrm{min}\left(1,{\frac{P(x^{\prime})}{P(x)}}{\frac{g(x\mid x^{\prime})}{g(x^{\prime}\mid x)}}\right)$

Where $P(x)$ is our desired distribution (in this case the Boltzmann distribution). This will also be the formula for $A(x',x)$ that we will be going with.

### The proposal distribution $g(x)$

This is a bit more tricky. In an ideal world we would like to know $\mu(x(t+\tau)|x(t))$, which is the probability of ne next state $x(t+\tau)$ given our current state $x(t)$. This would obviously propose the best possible (i.e. most probable) new states.

 We could theoretically perform many simulations starting at $x(t)$ and compute our probability in this way, but of course we have better options. The entire rest of the paper will be about finding a good approximation for this specific distribution.

> Even though this will just be an approximation, the final result of our MH will still converge in the correct distribution. This is because as previously mentioned, the proposal distribution (unlike the acceptance distribution) does not directly impact the final distribution

## Conditional normalizing flows

Just as with the Monte Carlo algorithm, most of the groundwork has already been laid out for us, and we mostly have to perform adjustments to the existing basis. Conditional normalizing flows (which are a special case of regular normalizing flows) are designed for the sole purpose of modeling some conditional distribution $p(x|y)$, so they are a perfect fit for our scenario!

Again, I would like to introduce the algorithm with the help of an illustration. This is a regular normalizing flow, but it can be easily adapted
(bold letters represent random variable vectors).

The general idea of normalizing flows is:
1. Start with a random variable vector $\mathbf{z}_0$ distributed by some simple base distribution $p_0$, e.g. a Gaussian distribution.
2. Apply a transformation $f$ (with some special properties) to the random variable vector, so that it is now distributed by some new $p_1$
3. Repeat this until your random variable vector $\mathbf{z}_K$ is distributed in the way you desire ($=\mathbf{x}$)

![Normalizing Flow](assets/normalizing-flow.png)

[Image source](https://lilianweng.github.io/posts/2018-10-13-flow-models/)

Obviously, with known $f_i$, this would easily allow us to randomly sample an easy distribution like the Gaussian, apply our transformations, and receive a sample that is sampled accordingly to our desired distribution $\mu(x(t+\tau)|x(t))$!

What's left for us is to actually find the functions $f_i$ that are responsible for the transformation, and make sure they fulfil all of the required properties (e.g. be easily computable). The shape and parametrization of these functions will be the topic of the next main step, which will conclude the whole setup for our MH algorithm.

> Note the similarity of normalizing flows with other layer-based models like neural networks and transformers. Much like in these models, having multiple layers allows for expressing more complex patterns and relationships while keeping the transformations between layers relatively simple

---

The reason this kind of transformation (from one random variable vector into another) is even possible, is thanks to (a generalized version of) the **change of variable theorem**,  which goes as follows:

Given $\mathbf{z} \sim \pi(\mathbf{z})$ and invertible $f$, you can create a new random variable vector $\mathbf{x} = f(\mathbf{z})$. The distribution of $\mathbf{x}$ will then be $p(\mathbf{x}) = \pi(f^{-1}(\mathbf{x})) \cdot \mathcal{J}$

So in our case, $\mathbf{x}$ always represents the random variable vector of the next layer (if $\mathbf{z}$ is the random variable vector of the current layer). $\mathcal{J}$ is the determinant of the Jacobian matrix of $f^{-1}$. From this we have two requirements for the functions $f_i$, which is that
1. The Jacobian determinant of $f^{-1}$ is easily computable
2. $f$ is easily invertible 

I highly recommend [Lilian Weng's blogpost](https://lilianweng.github.io/posts/2018-10-13-flow-models/) if you are interested in more mathematical details like proofs why our normalizing flow architecture actually fulfils these properties.

---

So let's apply the normalizing flow model to our problem.
Remember, we start with $x(t)$ and we want to end up in a sample from $\mu(x(t+\tau)|x(t))$

- Our random variable vector is made up by the two latent random variables, $z^p$ and $z^v$, which at the beginning are both sampled from the standard Gaussian:

$$z^p \sim \mathcal{N}(0,1) \quad z^v \sim \mathcal{N}(0,1)$$

- Our full flow $f_\theta = f_0 \circ ... \circ f_{K-1}$ is the transformation that takes $z^p$ and $z^v$ to their values in the resulting molecule $x(t+\tau)$, so  $z^p \rightarrow x^p(t+\tau)$ and $z^v \rightarrow x^v(t+\tau)$. The flow uses the beginning state of $x$ as starting info, so: $$x^{p}{\big(}t+\tau{\big)},x^{v}{\big(}t+\tau{\big)}:=f_{\theta}{\big(}z^{p},z^{v};x^{p}{\big(}t{\big)},x^{v}{\big(}t{\big)}{\big)}$$
 We can actually make our lives easier by treating $x^v$ as a *non-physical auxiliary variable*, rather than trying to accurately compute it. This is because in the final result (thinking back at the protein folding example), the velocities are not really of much interest.
 
  We end up with the flow:$$x^{p}{\big(}t+\tau{\big)},x^{v}{\big(}t+\tau{\big)}:=f_{\theta}{\big(}z^{p},z^{v};x^{p}{\big(}t{\big)}\phantom{,x^{v}{\big(}t{\big)}}{\big)}$$ Now instead of using the information $x^{v}{\big(}t{\big)}$ to be able to infer our new velocities $x^{v}{\big(}t+\tau{\big)}$, we simply randomly sample them from the Gaussian distribution.
  
  The simplification we make here will also impact our final Boltzmann distribution, which will from now on be called $\mu_{aug}$

## Finding functions $f$

As mentioned earlier, the functions $f$ need to exhibit some specific properties, meaning there is really just a subset of function families that make sense here. Again, some work has been done here before, and the model architecture we will choose is called **RealNVP**

The characteristics of RealNVP are:
1. It uses uses scaling and shifting transformations for $f$. Intuitively these also make sense to be easily invertible.
2. The transformations $f$ only transform part of the random variable vector in each step, allowing for an alternating pattern in the flow layers. In each step, the parameters of the scale and shift are given by the untransformed part of the random variable vector.

Our specific setup for RealNVP is shown in this picture:
![RealNVP](assets/realnvp.png)

These will be the first three layers of the model, annotated with the general transformation formulas. You can observe the properties of RealNVP here ($\odot$ is the element-wise product):

1. The transformations are of the "scale-and-shift" form (oversimplified here) $$z^p_{new} = s(z^v) \odot z^p + t(z^v)$$ $$z^v_{new} = s(z^p) \odot z^v + t(z^p)$$ Where $s$ and $t$ represent scaling and transforming respectively.
2. Each layer updates either the $z^p_i$ part of our random variable vector, or the $z^v_i$ part. The parameters of our scales and shifts are always the part of the random variable vector that is not being transformed

> As explained earlier, we have dropped $x^v(t)$ from our dependencies of the flow, hence our scaling and transforming is only conditioned on $x^p(t)$. Also, all the indexes $\ell$ are going to be left out for better clarity.

There is just one more question left. How do we know the scaling and transformation factors for $s$ and $t$?

### Atom transformers

As it is so common with current ML papers, we cannot get around using transformers here - both $s$ and $t$ are realized in the form of transformers. This makes obvious sense when we think about the molecules as sequences of atoms, where each atom corresponds to one input token.

Let's analyze the structure of the transformer that outputs a translation of $z^v$ (in the above image, this would be the second part of the first transition). All of the other transformers work in an analogue way.

1. Our inputs for this specific transformer are $z^p$ and $x^p(t)$
2. For each atom of the molecule we create the concatenated token $a_i := [x^p_i(t), h_i, z^v_i]^\intercal$, where $h_i$ is an embedding vector representing the atom type.
3. The token sequence $a_0, ..., a_N$ (given N atoms in the molecule) first passes through an MLP (or feedforward layer)
4. It is then passed repeadedly through the usual transformer layers (self-attention, normalizing)
5. Finally, we get back our tokens in the shape we want them to be, by passing through another MLP.

So far this is pretty standard transformer architecture, there are a few quirks though

1. It does not really make sense to give the atoms of the molecule an ordering (unlike e.g. with words in a sentence), so the positional encoding that is usually attached to each input token is left out
2. Usually in the self-attention layers, we use the dot product to compute how much attention each token should pay to each other token. We can save a significant amount of computing cost by simply assuming that atoms should usually pay more attention to the atoms that are close in position, and less attention to atoms that are far away. For this reason, an attention weight $w_{ij}$ is used, which is essentially a normalized distance from atom $a_i$ to atom $a_j$.

This graphic from the paper nicely summarizes the entire normalizing flow structure:

![Whole Architecture](assets/architecture.png)
[Image source](https://arxiv.org/pdf/2302.01170)

Note the diagonal lines signifying where the two rightmost subcomponents fit into the flow. The flow starts with $x^p(t)$ and $z^p, z^v$ sampled normally, and ends with $x^p(t+\tau), \ x^v(t+\tau)$. You can easily see the RealNVP approach in the alternating scaling and transformation of our inputs, as well as the transformer architecture which was just described.

## Training

We have finally set everything up, and now it's time to perform training on the parameters $\theta$ for our normalizing flow to approximate $\mu(x(t+\tau)|x(t))$.
 
For the dataset, sequences $\mathcal{T}_i = (x(0),x(\tau),x(2\tau),...)$ were generated for different peptides using a traditional MD Library.

Since the training data is really what determines the scope of the possible input data, it can be assumed the model generalizes well for all kinds of peptides, but possibly not too well for completely different molecular systems.

For the Loss function, there are actually two objectives we are interested in optimizing:
1. The deviation from the dataset, which is the standard log likelihood: $$\mathcal{L}_{\mathrm{lik}}(\theta):=\frac{1}{K}\sum_{k=1}^{K} \text{log} \ p_{\theta}(x^{(k)}(t+\tau)\vert x^{(k)}(t))$$

2. The acceptance in MH. Recall that: "A good proposal distribution will lead to better states being generated, and thus more states being accepted". \
To maximize our acceptance, we need to maximize the term (the big fraction has been shortened to $r_{\theta}$ for brevity) $$A(x^{\prime},x)=\mathrm{min}\left(1,r_{\theta}(x,x')\right)$$ Which is equivalent to maximizing $r_{\theta}$ directly. This results in the loss function: $$\mathcal{L}_{\mathrm{acc}}(\theta):=\frac{1}{K}\sum_{k=1}^{K}\log r_{\theta}(x^{(k)}(t),{x'}_{\theta}^{(k)}(t+\tau))$$ The issue here is that in order to please the acceptance distribution, only conservative changes in state will be proposed, which is not productive either (exploration vs. exploitation). To counteract this, we instead use a Loss function very similar to $\mathcal{L}_{\mathrm{lik}}(\theta)$: $$L_{\text{ent}}(\theta) = -\frac{1}{K} \sum_{k=1}^K \log p_{\theta}({x'}_{\theta}^{(k)}(t + \tau) | x^{(k)}(t))$$ This can be interpreted as the negative log-likelihood of the proposed transitions according to the model, allowing for more exploration in our transitions.


## Putting everything together

We really have everything we need now. I would like to reiterate that in the core of all of these complicated computations, we still use basic Metropolis Hastings. Here is the final setup:

We want to sample from our modified Boltzmann distribution $\mu_{aug}$, which now ignores the physical impact of the atom velocities, and instead samples them from a Gaussian distribution:

$$\mu_{\mathrm{aug}}(x^{p},x^{v})\propto\exp\left(-\frac{U(x^{p})}{k_{B}T}\right)\mathcal{N}(x^{v};0,I)$$

We start from an initial state of our molecule $x_0 = (x_0^p, x_0^v)$.
We propose a new state $x_1 \sim p_\theta( \ \cdot \mid x^p_1 )$.

We accept the new state with the probability given by $A(x',x)$, where

$$A(x,x')=\mathrm{min}\left(1,\,\frac{\mu_{\mathrm{aug}}(x')p_{\theta}(x\mid x'^{p})}{\mu_{\mathrm{aug}}(x)p_{\theta}(x'\mid\,x^{p})}\right)$$

This is simply the acceptance ratio given in the beginning, with our own values for $P(x)$ and $g(x)$

After each step, we also perform a Gibbs sampling update
$\left(x_{m}^{p},x_{m}^{v}\right)\leftarrow\;\left(x_{m}^{p},\epsilon\right),\quad\epsilon\sim\mathcal{N}(0,I)$. This essentially "completes" our new state, by estimating the most probable velocity $x^v_m$ based on the current position, since the velocity was originally just normally distributed.

This process is repeated until we achieve our desired accuracy of $\mu_{aug}$, which took the authors around 30 minutes on 4 NVIDIA A-100 GPUs (depending on the experiment).

# Experiments

Timewarp was tested on different kinds of molecular systems, which challenged different aspects of its performance.
- For molecular systems with a large amount of different molecules, the generizability of Timewarp was put to the test. The training set cannot cover everything (sometimes just 1% of the possible molecules), and even so Timewarp still performed well in these tests
- For molecular systems with metastable states (e.g. folded states) that are hard to reach, the exploration ability was tested. For some systems, Timewarp performed better than traditional MD, for some worse. In the median though, Timewarp was able to discover all metastable states 5 times faster (and up to 33 times faster!) than traditional MD. Additionally, a variant of Timewarp was deployed that only focuses on exploration of the statespace (at the expense of accuracy), which did find these difficult states, which could then later be confirmed with regular Timewarp.
- For all datasets there was a relatively low acceptance rate of the MH steps between 0.03% - 2%

Taking a closer look at one of the experiments, we can see a clear comparison with traditional MD.
![Experiments](assets/experiments.png)
[Image source](https://arxiv.org/pdf/2302.01170) (with custom modifications)

These are the results of running timewarp on an Alanine dipeptide molecule.

1. Shows Ramachandran plot, which visualizes the $\phi$ and $\psi$ angles. Simply put, these are the angles in the molecule that determine most of its full structure. Brighter areas indicate more energetically favored states (or metastable states). As can be seen, the results are very similar to traditional MD.

2. Shows the free energy for given $\phi$ or $\psi$ angles. Again these graphs line up closely with traditional MD. Some slight jittering can be observed in the higher values of the graph, which can be explained by the fact that Timewarp did not spend a lot of time sampling in these areas (higher energy is unfavorable, and so Timewarp looked for different states)

3. Shows the conditional distributions for the state proposals $\mu(x(t+\tau)\vert x(t))$ (left), $p_\theta(x(t+\tau)\vert x(t))$ (right), with their initial state $x(t)$ (red cross). Again, the proposals made by MD align very closesly with those made by traditional MD. This also shows that our simplification of dropping the molecule velocities did not impact this result significantly.



# Conclusion

I think Timewarp is a fantastic step in making molecular simulations more efficient and more transferable. It can be easily adapted to fit different purposes, like modifying the MH step to instead run an exploration algorithm as mentioned earlier, and I think this flexibility will be of central importance when integrated into bigger systems.

It is apparent that throughout the entire way, the authors aimed to choose options that allow the final product to be transferable (like representing molecules in cartesian coordinates rather than all-atom resolution, or choosing normalizing flows which have proven in the past to be great for these purposes). And I think this is a perfect application of deep learning methods in this sense.

Even though the acceptance rate in MH is quite low, Timewarp still achieves a very impressive speedup over traditional MD, while still keeping up regarding results.

Future research could aim into improving this acceptance rate, or even aim for transferability in larger molecular systems.

---

Thank you for reading my blog post, and I hope you enjoyed learning about this paper as much as I did!

# References

Leon Klein, Andrew Y. K. Foong, Tor Erlend Fjelde, Bruno Mlodozeniec, Marc Brockschmidt, Sebastian Nowozin, Frank Noé, Ryota Tomioka. (2023). Timewarp: Transferable Acceleration of Molecular Dynamics by Learning Time-Coarsened Dynamics. *arXiv preprint*. Available at: [https://arxiv.org/abs/2302.01170](https://arxiv.org/abs/2302.01170)

Weng, Lilian. (2018). Flow-based Deep Generative Models. Available at:[https://lilianweng.github.io/posts/2018-10-13-flow-models/](https://lilianweng.github.io/posts/2018-10-13-flow-models/)


Seung-Seop Jin, Heekun Ju, Hyung-Jo Jung. (2019). Adaptive Markov chain Monte Carlo algorithms for Bayesian inference: recent advances and comparative study. Available at [http://dx.doi.org/10.1080/15732479.2019.1628077](http://dx.doi.org/10.1080/15732479.2019.1628077)
