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
print("\nSum over sp for s=0, a=0:", np.sum(p_theta[0, 0]))  # Should be ~1.0


# %% [code]
def mle_loss(theta, kappa, n_states, n_actions, p_true, r_true):
    """
    theta is the flat parameter vector (n_r + n_p elements)
    kappa is the L2 norm bound
    n_states is the number of states
    n_actions is the number of actions
    p_true is the true transition probabilities
    r_true is the true reward matrix

    """
    r_theta, p_theta = theta_to_model(theta, n_states, n_actions, kappa)

    kl_loss = 0.0
    for s in range(n_states):
        for a in range(n_actions):
            for sp in range(n_states):
                # ignore zero probabilities
                if p_true[s, a, sp] > 0:
                    kl_loss += p_true[s, a, sp] * np.log(
                        # the +1e-10 is to avoid log(0)
                        p_true[s, a, sp] / (p_theta[s, a, sp] + 1e-10)
                    )

    r_mse = 0.0
    for s in range(n_states):
        for a in range(n_actions):
            r_mse += (r_true[s, a] - r_theta[s, a]) ** 2

    loss = kl_loss + r_mse

    return loss

loss = mle_loss(theta, kappa, n_states, n_actions, p_true, r_true)
print(loss)


# %% [code]
def omd_loss(theta, kappa, n_states, n_actions, p_true, r_true):

    r_theta, p_theta = theta_to_model(theta, n_states, n_actions, kappa)

    Q_star, iterations_done = value_iteration(p_theta, r_theta)

    loss = 0.0

    for s in range(n_states):
        for a in range(n_actions):
            B_true_Q = 0.0
            for sp in range(n_states):
                exp_total = 0.0
                for ap in range(n_actions):
                    exp_total += np.exp(Q_star[sp, ap] / 0.1)
                logsumexp_sp = 0.1 * np.log(exp_total)

                B_true_Q += p_true[s, a, sp] * logsumexp_sp

            loss += (B_true_Q - Q_star[s, a]) ** 2
    return loss


loss = omd_loss(theta, kappa, n_states, n_actions, p_true, r_true)
print(loss)


# %% [code]
def optimize(loss_fn, theta_init, kappa, n_states, n_actions, p_true, r_true, learning_rate=0.1, inner_loop_steps=100, grad_step_size = 1e-5,  max_iterations=1000):

    theta = theta_init.copy()
    losses = []

    for outer_loop in range(max_iterations):

        grad = np.zeros_like(theta)
        for i in range(len(theta)):
            theta_plus = theta.copy()
            theta_minus = theta.copy()
            theta_plus[i] += grad_step_size
            theta_minus[i] -= grad_step_size

            loss_plus = loss_fn(theta_plus, kappa, n_states, n_actions, p_true, r_true)
            loss_minus = loss_fn(theta_minus, kappa, n_states, n_actions, p_true, r_true)

            grad[i] = (loss_plus - loss_minus) / (2 * grad_step_size)

        theta -= learning_rate * grad

        losses.append(loss_fn(theta, kappa, n_states, n_actions, p_true, r_true))

        if outer_loop % 20 == 0:
            print(f"iter {outer_loop}: loss = {losses[-1]:.4f}, ‖θ‖ = {np.linalg.norm(theta):.4f}")

        if np.linalg.norm(theta) > kappa:
            theta = (kappa / np.linalg.norm(theta)) * theta

    print(f"final:   loss = {losses[-1]:.4f}, ‖θ‖ = {np.linalg.norm(theta):.4f}")
    return theta, losses


 # MLE
theta_init = np.random.randn(12)
theta_mle, losses_mle = optimize(
   mle_loss, theta_init, kappa=0.5, n_states=2, n_actions=2,
   p_true=p_true, r_true=r_true, learning_rate=0.01, max_iterations=100
)

print(f"MLE final loss: {losses_mle[-1]:.4f}")

# OMD
theta_init = np.random.randn(12)
theta_omd, losses_omd = optimize(
    omd_loss, theta_init, kappa=0.5, n_states=2, n_actions=2,
    p_true=p_true, r_true=r_true, learning_rate=0.01, max_iterations=100
)

print(f"OMD final loss: {losses_omd[-1]:.4f}")


# %% [code]
# Get returns for MLE
r_mle, p_mle = theta_to_model(theta_mle, n_states, n_actions, kappa=0.5)
Q_mle, _ = value_iteration(p_mle, r_mle)
J_mle = expected_return(Q_mle, p_true, r_true)
print(f"MLE return: {J_mle:.4f}")

# Get returns for OMD
r_omd, p_omd = theta_to_model(theta_omd, n_states, n_actions, kappa=0.5)
Q_omd, _ = value_iteration(p_omd, r_omd)
J_omd = expected_return(Q_omd, p_true, r_true)
print(f"OMD return: {J_omd:.4f}")

np.random.seed(0)
theta_init = np.random.randn(12)
theta_omd, _ = optimize(omd_loss, theta_init, 1.0, n_states, n_actions, p_true, r_true, learning_rate=0.01, max_iterations=100)
r_o, p_o = theta_to_model(theta_omd, n_states, n_actions, 1.0)
Q_o, _ = value_iteration(p_o, r_o)
print("r_θ:", r_o)
print("p_θ:", p_o)
print("Q:", Q_o)
print("policy:", get_policy_from_Q(Q_o))


# %% [code]
kappa_sweep = False
if kappa_sweep:
    # kappa sweep, multiple seeds, same init for MLE and OMD
    kappas = [0.1, 0.5, 1.0, 2.0, 5.0]
    seeds = [0, 1, 2, 3, 4]

    results = {"mle": {k: [] for k in kappas}, "omd": {k: [] for k in kappas}}

    for kappa in kappas:
        for seed in seeds:
            np.random.seed(seed)
            theta_init = np.random.randn(12)

            # MLE
            theta_mle, _ = optimize(
                mle_loss, theta_init, kappa, n_states, n_actions,
                p_true, r_true, learning_rate=0.01, max_iterations=100,
            )
            r_m, p_m = theta_to_model(theta_mle, n_states, n_actions, kappa)
            Q_m, _ = value_iteration(p_m, r_m)
            J_m = expected_return(Q_m, p_true, r_true)
            results["mle"][kappa].append(J_m)

            # OMD (same init as MLE)
            theta_omd, _ = optimize(
                omd_loss, theta_init, kappa, n_states, n_actions,
                p_true, r_true, learning_rate=0.01, max_iterations=100,
            )
            r_o, p_o = theta_to_model(theta_omd, n_states, n_actions, kappa)
            Q_o, _ = value_iteration(p_o, r_o)
            J_o = expected_return(Q_o, p_true, r_true)
            results["omd"][kappa].append(J_o)

    # summary table
    print("\n" + "=" * 60)
    print(f"{'kappa':>8} | {'MLE mean':>10} {'MLE std':>10} | {'OMD mean':>10} {'OMD std':>10}")
    print("-" * 60)
    for kappa in kappas:
        mle_arr = np.array(results["mle"][kappa])
        omd_arr = np.array(results["omd"][kappa])
        print(f"{kappa:>8.2f} | {mle_arr.mean():>10.4f} {mle_arr.std():>10.4f} | {omd_arr.mean():>10.4f} {omd_arr.std():>10.4f}")
    print("=" * 60)


