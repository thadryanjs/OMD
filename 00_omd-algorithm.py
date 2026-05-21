# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---


# %% [code]
import numpy as np

p_true = np.array([
    # state 0
    # a=0: s'=0 with prob 1.0
    [[1.0, 0.0],
    # a=1: s'=1 with prob 1.0
    [0.0, 1.0]],

    # state 1
    # a=0: s'=0 with prob 1.0
    [[1.0, 0.0],
    # a=1: s'=1 with prob 1.0
    [0.0, 1.0]]
])

r_true = np.array([
    # state 0
    # a0 -> 1.0, a1 -> 0.0
    [1.0, 0.0],
    # state 1
    # a0 -> 0.0, a1 -> 2.0
    [0.0, 2.0]
])

# set seed
np.random.seed(1234)
# discounted expected reward (starts as a random guess)
Q = np.random.rand(2, 2)


# %% [code]
def soft_bellman_opt_explicit(Q, p, r, gamma=0.9, alpha=0.1):
    """
    Q is the current value function
    p is the transition matrix
    r is the reward matrix
    gamma is the discount factor
    alpha is the softmax temperature
    """
    n_states = p.shape[0]
    n_actions = p.shape[1]
    # B_Q - estimated discounted expected reward
    B_Q = np.zeros((n_states, n_actions))
    for s in range(n_states):
        for a in range(n_actions):
            # s prime
            for sp in range(n_states):
                exp_total = 0.0
                for ap in range(n_actions):
                    exp_total += np.exp(Q[sp, ap] / alpha)
                logsumexp_sp = alpha * np.log(exp_total)
                # add the accumulated reward for each s prime
                B_Q[s, a] += p[s, a, sp] * logsumexp_sp
            # the actual computation
            B_Q[s, a] = r[s, a] + gamma * B_Q[s, a]
    return B_Q

# breakpoint()
soft_bellman_opt_explicit(Q, p_true, r_true)


# %% [code]
def value_iteration(p, r, gamma=0.9, alpha=0.1, tol=1e-6, max_iter=1000):
    """
    p is the transition matrix
    r is the reward matrix
    gamma is the discount factor
    alpha is the softmax temperature
    tol is the tolerance for convergence
    max_iter is the maximum number of iterations
    """
    n_states = p.shape[0]
    n_actions = p.shape[1]
    Q = np.zeros((n_states, n_actions))

    for i in range(max_iter):
        B_Q = soft_bellman_opt_explicit(Q, p, r, gamma, alpha)
        delta = np.max(np.abs(Q - B_Q))
        Q = B_Q
        if delta < tol:
            return Q, i+1
    # using i+1 here give possible unbound var war
    return Q, max_iter


Q_star, iterations_done = value_iteration(p_true, r_true)


# %% [code]
def get_policy_from_Q(Q, alpha=0.1):
    """
    Q is the current value function
    alpha is the softmax temperature
    """
    n_states = Q.shape[0]
    n_actions = Q.shape[1]

    pi = np.zeros((n_states, n_actions))

    for s in range(n_states):
        exp_Q = np.zeros(n_actions)
        for a in range(n_actions):
            exp_Q[a] = np.exp(Q[s, a] / alpha)

        exp_total = 0.0
        for a in range(n_actions):
            exp_total += exp_Q[a]

        for a in range(n_actions):
            pi[s, a] = exp_Q[a] / exp_total

    return pi

pi_star = get_policy_from_Q(Q)
print(pi_star)


# %% [code]
def expected_return(Q, p, r, alpha=0.1, rho0=None):
    """
    Q is the action-value function
    p is the transition matrix
    r is the reward matrix
    gamma is the discount factor
    alpha is the softmax temperature
    rho0 is the initial state distribution
    """

    n_states = p.shape[0]
    n_actions = p.shape[1]

    if rho0 is None:
        rho0 = np.ones(n_states) / n_states

    pi = get_policy_from_Q(Q, alpha)

    J = 0.0
    for s in range(n_states):
        weighted_Q = 0.0
        for a in range(n_actions):
            weighted_Q += pi[s, a] * Q[s, a]
        J += rho0[s] * weighted_Q

    return J

J = expected_return(Q_star, p_true, r_true)
print(J)


# %% [code]
def theta_to_model(theta, n_states, n_actions, kappa=1.0):
    """
    Args:
        theta: flat parameter vector (n_r + n_p elements)
        n_states: number of states
        n_actions: number of actions
        kappa: L2 norm bound (controls misspecification)

    Returns:
        r_theta: [n_states, n_actions] reward matrix
        p_theta: [n_states, n_actions, n_states] transition probabilities

    Structure:
        r_θ:          p_θ_logits:
        [r00, r01]    [p000, p001]
        [r10, r11]    [p010, p011]
                      [p100, p101]
                      [p110, p111]

    Flattened into theta:
        [r00, r01, r10, r11,  p000, p001, p010, p011, p100, p101, p110, p111]
        │──── 4 params ────│  │──────────── 8 params ───────────────────────│
       """
    if np.linalg.norm(theta) > kappa:
        theta = (kappa / np.linalg.norm(theta)) * theta

    n_r = n_states * n_actions
    n_p = n_states * n_actions * n_states

    r_theta = theta[:n_r].reshape((n_states, n_actions))
    p_logits = theta[n_r:n_r+n_p].reshape((n_states, n_actions, n_states))

    p_theta = np.exp(p_logits) / np.sum(
        np.exp(p_logits), axis=2, keepdims=True
    )

    return r_theta, p_theta

theta = np.random.randn(12)
kappa = 1.0
n_states = 2
n_actions = 2
r_theta, p_theta = theta_to_model(theta, n_states, n_actions, kappa)

print("r_θ:\n", r_theta)
print("\np_θ shape:", p_theta.shape)
print("\nSum over sp for s=0, a=0:", np.sum(p_theta[0, 0]))  # Should be 1.0
