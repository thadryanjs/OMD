# Nikishin et al. 2022 — OMD (Optimal Model Design) — Working Summary

> Control-Oriented Model-Based RL with Implicit Differentiation (AAAI-22)

## TL;DR

OMD is a **control-oriented** MBRL method that learns model parameters $\theta$ by **directly differentiating expected return through a learned $Q$-function** that is treated as an *implicit* function of $\theta$. The model is *not* trained with maximum likelihood. Instead, the inner loop fits a $Q$-network to the Bellman fixed point of the *current* model, and the outer loop pushes $\theta$ in whichever direction reduces the Bellman error against the *true* environment — using the implicit function theorem to backprop through the inner optimization.

Key payoff: under **model misspecification** (limited capacity, distractor states, etc.) OMD beats MLE/MBPO-style models, even though OMD's models can have *worse-than-random* prediction MSE while still producing useful targets for $Q$-learning.

---

## Notation

| Symbol | Meaning |
|---|---|
| $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \gamma, p, r, \rho_0)$ | True MDP |
| $p_\theta, r_\theta$ | Learned model dynamics and reward, parameters $\theta$ |
| $Q_w$ | Q-network with parameters $w$; $\bar{w}$ is its target/EMA copy |
| $\alpha$ | Soft-Bellman / softmax temperature (small, e.g. $0.01$) |
| $\varphi(\theta) = w^*$ | Implicit function: optimal $w$ given fixed $\theta$ |
| $K$ | Inner-loop steps per outer step |

---

## Core formulas

### 1. Soft Bellman optimality operator (model-induced)

The model's Bellman operator with a smooth (log-sum-exp) max:

$$
B^\theta Q(s, a) \triangleq r_\theta(s, a) + \gamma \, \mathbb{E}_{s' \sim p_\theta(\cdot | s, a)} \! \log \! \sum_{a'} \exp Q(s', a').
\tag{3}
$$

The true-environment counterpart $B$ uses $r$ and $p$ instead of $r_\theta$ and $p_\theta$.

(With temperature: replace $\log \sum_{a'} \exp Q(s', a')$ by $\alpha \log \sum_{a'} \exp \tfrac{1}{\alpha} Q(s', a')$. As $\alpha \to 0$ this becomes $\max_{a'} Q(s', a')$.)

### 2. Policy from Q (softmax / Boltzmann)

$$
\pi_Q(a \mid s) = \frac{\exp Q(s, a)}{\sum_{a'} \exp Q(s, a')}.
$$

### 3. Tabular OMD problem (exact)

Two equivalent constrained problems; the second (Bellman-error) is what the deep version uses.

Policy form (eq. 2):

$$
\max_{Q, \theta} J(\pi_Q) \quad \text{s.t.} \quad Q(s, a) = B^\theta Q(s, a) \;\; \forall s, a.
$$

Value form (eq. 6):

$$
\min_{Q, \theta} \; L^{\text{true}}(Q) \triangleq \sum_{s, a} \big( Q(s, a) - B Q(s, a) \big)^2 \quad \text{s.t.} \quad Q(s, a) = B^\theta Q(s, a) \;\; \forall s, a.
$$

The inner constraint says $Q$ is the fixed point of the *model's* operator. The outer objective measures how good that $Q$ is under the *true* operator.

### 4. Implicit function theorem (for getting $\partial w^* / \partial \theta$)

Let $f(\theta, w) = \mathbf{0}$ define $w^* = \varphi(\theta)$. If $\partial f / \partial w$ is invertible at $(\theta, w^*)$:

$$
\frac{\partial \varphi(\theta)}{\partial \theta}
= -\Big( \tfrac{\partial f(\theta, w^*)}{\partial w} \Big)^{-1} \!\cdot\! \tfrac{\partial f(\theta, w^*)}{\partial \theta}
\;\Big|_{w^* = \varphi(\theta)}.
\tag{7}
$$

In OMD, the constraint that defines $\varphi$ at the deep scale is the first-order optimality condition (eq. 12):

$$
f(\theta, w) \;=\; \frac{\partial L(\theta, w)}{\partial w} \;=\; \mathbf{0},
\tag{12}
$$

with the inner-loop loss (eq. 11):

$$
L(\theta, w) \;\triangleq\; \mathbb{E}_{s, a} \big[ Q_w(s, a) - B^\theta Q_{\bar w}(s, a) \big]^2.
\tag{11}
$$

### 5. Outer-loop ("true") objective for deep OMD

Bellman error against the *true* model — used **only** to take the gradient step on $\theta$. $w$ is not optimized through this loss.

$$
L^{\text{true}}(w) \;\triangleq\; \mathbb{E}_{s, a} \big[ Q_w(s, a) - B Q_{\bar w}(s, a) \big]^2.
\tag{13}
$$

### 6. The OMD outer-loop gradient (the thing you actually code)

After applying the IFT and the standard practical approximation that replaces the inverse Jacobian $\big(\partial f / \partial w\big)^{-1}$ with the identity (a simplification used in Rajeswaran-Mordatch-Kumar 2020; the paper reports it works as well or better in their setting):

$$
\boxed{\;
\frac{\partial L^{\text{true}}(\theta)}{\partial \theta}
\;\approx\; -\,\underbrace{\frac{\partial L^{\text{true}}(w^*)}{\partial w}}_{\text{grad of true Bellman error wrt } w}
\,\cdot\,
\underbrace{\frac{\partial^2 L(\theta, w^*)}{\partial \theta \, \partial w}}_{\text{mixed partial of inner loss}}
\;\Big|_{w^* = \varphi(\theta)}
\;}
\tag{14}
$$

Equivalently, this is `vjp(grad_w L_true, theta)` evaluated at $w^* = \varphi(\theta)$, which JAX-style autodiff can compute directly.

---

## Algorithm 1 — Model-Based RL with OMD

**Input:** initial $w$, $\theta$; empty replay buffer $\mathcal{D}$.

```
repeat:
    Set s = current state.
    Sample a ~ softmax(Q_w(s, ·)).
    Apply a in the TRUE env  →  r = r(s, a),  s' ~ p(· | s, a).
    Append (s, a, s', r) to D.

    for i = 1 ... K:                           # inner loop
        Sample (s, a) from D.
        Use the MODEL to get  r̂ = r_θ(s, a),  ŝ' ~ p_θ(· | s, a).
        Update w to minimize L(θ, w) =  E[(Q_w(s,a) - B^θ Q_w̄(s,a))²]
                                       (target uses model-generated r̂, ŝ')

    Update θ via the gradient in eq. (14)      # outer loop, uses TRUE r, s'
until max interactions
```

Two pieces of detail worth flagging:

- The **inner loop targets** are built from the *model's* $r_\theta$ and $p_\theta$ — i.e. $w$ chases the Bellman fixed point of the model.
- The **outer loop gradient** uses the *true* environment's $r$ and $s'$ (collected during interaction and stored in $\mathcal{D}$). This is the only place where the true model "leaks" into the model-update path; that leak is exactly the control signal that tells $\theta$ "make $Q_w$ a better approximation of the *true* $Q^*$."

---

## Practical implementation notes

- **Inner-loop warm start.** Don't reinitialize $w$ each outer step — reuse it from the previous outer iteration so $K$ small (paper uses just a handful of inner steps) is enough to keep $w \approx \varphi(\theta)$.
- **Inverse-Jacobian approximation.** They replace $\big(\partial^2 L / \partial w^2 \big)^{-1}$ with $I$. The paper reports they did *not* see benefits from using the actual Hessian inverse, consistent with Rajeswaran et al. 2019 and Lorraine et al. 2020.
- **Target network $\bar w$.** EMA-tracked copy of $w$, standard DQN-style stability trick. Used inside both $L$ (eq. 11) and $L^{\text{true}}$ (eq. 13).
- **Double Q-learning.** Used in their CartPole runs; omitted from equations for clarity.
- **Outer-loop loss choice: $L^{\text{true}}$ vs $J$.** Both work; they found $L^{\text{true}}$ converges with fewer samples and *still* corresponds to maximizing entropy-regularized return. Use $L^{\text{true}}$.
- **HalfCheetah inner optimizer.** They use SAC (`jaxrl` impl, https://github.com/ikostrikov/jaxrl) as the inner Q-/policy-fitter, with default config. The MLE comparator with this setup is essentially MBPO without ensembling.
- **Action sampling.** Softmax over $Q_w$ (or, for continuous control, the SAC actor).
- **Discount factor $\gamma$ for tabular toy MDP:** 0.9. **Softmax temperature $\alpha$:** 0.01.

---

## Why this works (one-paragraph intuition)

If the model class is too small to represent $p$ exactly, MLE will spend its limited capacity matching the bits of $p$ that minimize KL — those bits may have nothing to do with the bits that determine $Q^*$. OMD instead asks: "what model, when I solve its own Bellman equation, gives me a $Q$ that predicts true returns well?" and gradient-descends $\theta$ on exactly that question. The set of $\theta$ that produce the same induced Bellman operator on $Q^*$ — the **$Q^*$-equivalence class** $\Theta_{Q^*}$ — is large and includes simple/deterministic models even when the true MDP is stochastic (see Figure 2). OMD is free to land anywhere in this class, including on a model that looks nothing like the true environment but still drives the agent to optimal action selection.

---

## Approximation bound (Theorem 2, stated only)

Let $\hat Q_{\text{MLE}}, \hat Q_{\text{OMD}}$ be the fixed points of the model-induced Bellman operators for the trained MLE / OMD models.

- **MLE:** if $\max_{s,a} \| p(\cdot | s, a) - \hat p(\cdot | s, a) \|_1 = \epsilon_p$, $\max_{s,a} | r - \hat r | = \epsilon_r$, $r \in [0, r_{\max}]$:

$$
\max_{s, a} \big| Q^*(s, a) - \hat Q_{\text{MLE}}(s, a) \big| \;\leq\; \frac{\epsilon_r}{1 - \gamma} + \frac{\gamma \, \epsilon_p \, r_{\max}}{2 (1 - \gamma)^2}.
$$

- **OMD:** if $\max_{s,a} \big| B \hat Q_{\text{OMD}}(s, a) - B^{\hat\theta} \hat Q_{\text{OMD}}(s, a) \big| = \epsilon$:

$$
\max_{s, a} \big| Q^*(s, a) - \hat Q_{\text{OMD}}(s, a) \big| \;\leq\; \frac{\epsilon}{1 - \gamma}.
$$

OMD directly minimizes the quantity that appears in its bound; MLE minimizes $\epsilon_p, \epsilon_r$, which only indirectly bound $Q^*$ error and do so via a $1/(1-\gamma)^2$ factor.

---

## What this paper is *not* doing

- Not training the model with maximum likelihood at any point.
- Not using a model ensemble (unlike MBPO).
- Not learning a variance prediction (Gaussian model has fixed variance).
- Not propagating gradients through trajectories rolled out in the model — gradients flow only through the IFT relating $w^*$ to $\theta$ at a single time step.
- Not using Lagrangians (they remark this is an alternative; conjecture it's less stable in the non-tabular case).

---

## Hyperparameters and setup

| Setting | Value / choice |
|---|---|
| Tabular toy MDP | 2 states, 2 actions, $\gamma = 0.9$, uniform $\rho_0$, $\alpha = 0.01$ |
| Model misspec knob (tabular) | Norm bound $\kappa$ on $\theta$ via projection $\theta \leftarrow \tfrac{\kappa}{\|\theta\|}\theta$ if $\|\theta\| > \kappa$ |
| CartPole misspec knob | Number of hidden units in the model network |
| CartPole noise knob | Append $N$ standard-Gaussian distractor states to obs |
| HalfCheetah inner optimizer | SAC (jaxrl default config) |
| Seeds | 10 (CartPole), 5 (HalfCheetah) |
| Inner-loop steps $K$ | Hyperparameter (small; weights warm-started across outer steps) |
| Inverse Jacobian | Approximated as identity |

---

## Where OMD fits among baselines

- **MLE / MBPO-style.** Trains $p_\theta, r_\theta$ to predict next-state and reward. Optimal under perfect capacity, suboptimal when misspecified.
- **VEP (Grimm et al. 2020).** Trains $\theta$ so the *model-induced* Bellman operator matches the *true* Bellman operator for a fixed set $\Pi, \mathcal{V}$ of policies and value functions:

$$
\ell_{\text{VEP}}(\theta) = \sum_{\pi \in \Pi} \sum_{V \in \mathcal{V}} \sum_{s \in \mathcal{S}} \big( B_\pi V(s) - B^\theta_\pi V(s) \big)^2,
$$

with $B^\theta_\pi V(s) = \mathbb{E}_{a \sim \pi(\cdot|s),\, s' \sim p_\theta(\cdot|s,a)}[r_\theta(s,a) + \gamma V(s')]$. Tested by the original authors only in tabular and CartPole-like settings. Underperforms OMD in their distractor experiments.
- **OMD (this paper).** Effectively learns *one specific* $Q^*$-equivalent model — the one autodiff lands on — via implicit differentiation through the inner Q-fit.
