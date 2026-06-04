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


def stable_logsumexp_alpha(values, alpha):
    max_value = np.max(values)
    exp_total = 0.0
    for value in values:
        exp_total += np.exp((value - max_value) / alpha)
    return max_value + alpha * np.log(exp_total)


config = {
    "seed": 1234,
    "gamma": 0.9,
    "alpha": 0.01,
    "kappa": 1.0,
    "value_iteration_tol": 1e-6,
    "value_iteration_max_iter": 1000,
    "fd_step": 1e-5,
    "mle_learning_rate": 0.01,
    "mle_max_iterations": 100,
    "mle_print_every": 20,
    "inner_q_learning_rate": 0.01,
    "inner_q_max_iterations": 100,
    "inner_q_print_every": 20,
    "outer_learning_rate": 0.01,
    "outer_max_iterations": 100,
    "outer_grad_step": 1e-4,
    "outer_print_every": 5,
    "mdp_name": "complex",  # "paper" or "complex"
    "calibration_n_transitions": 1000,
    "calibration_alpha": 0.1,
    "convergence_window": 10,
    "convergence_eps": 1e-4,
}

p_paper = np.array([
    [[1.0, 0.0],
     [0.0, 1.0]],
    [[1.0, 0.0],
     [0.0, 1.0]],
])

r_paper = np.array([
    [1.0, 0.0],
    [0.0, 2.0],
])

p_complex = np.array([
    # state 0
    [[0.7, 0.2, 0.1],
     [0.1, 0.6, 0.3]],
    # state 1
    [[0.3, 0.4, 0.3],
     [0.5, 0.1, 0.4]],
    # state 2
    [[0.2, 0.5, 0.3],
     [0.4, 0.3, 0.3]],
])

r_complex = np.array([
    [1.0, 0.0],
    [0.5, 1.5],
    [0.0, 2.0],
])

if config["mdp_name"] == "paper":
    p_true = p_paper
    r_true = r_paper
elif config["mdp_name"] == "complex":
    p_true = p_complex
    r_true = r_complex
else:
    raise ValueError(f"Unknown mdp_name: {config['mdp_name']}")

print("MDP:", config["mdp_name"])
print("p_true shape:", p_true.shape)
print("r_true shape:", r_true.shape)

# set seed
np.random.seed(config["seed"])
# discounted expected reward (starts as a random guess)
Q = np.random.rand(p_true.shape[0], p_true.shape[1])


# %% [code]
def soft_bellman_opt_explicit(Q, p, r, gamma=0.9, alpha=0.01):
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
                logsumexp_sp = stable_logsumexp_alpha(Q[sp], alpha)
                # add the accumulated reward for each s prime
                B_Q[s, a] += p[s, a, sp] * logsumexp_sp
            # the actual computation
            B_Q[s, a] = r[s, a] + gamma * B_Q[s, a]
    return B_Q

# breakpoint()
soft_bellman_opt_explicit(Q, p_true, r_true)


# %% [code]
def value_iteration(p, r, gamma=0.9, alpha=0.01, tol=1e-6, max_iter=1000):
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


Q_star, iterations_done = value_iteration(
    p_true,
    r_true,
    gamma=config["gamma"],
    alpha=config["alpha"],
    tol=config["value_iteration_tol"],
    max_iter=config["value_iteration_max_iter"],
)


# %% [code]
def get_policy_from_Q(Q, alpha=0.01):
    """
    Q is the current value function
    alpha is the softmax temperature
    """
    n_states = Q.shape[0]
    n_actions = Q.shape[1]

    pi = np.zeros((n_states, n_actions))

    for s in range(n_states):
        max_value = np.max(Q[s])
        exp_Q = np.zeros(n_actions)
        for a in range(n_actions):
            exp_Q[a] = np.exp((Q[s, a] - max_value) / alpha)

        exp_total = 0.0
        for a in range(n_actions):
            exp_total += exp_Q[a]

        for a in range(n_actions):
            pi[s, a] = exp_Q[a] / exp_total

    return pi

pi_star = get_policy_from_Q(Q, alpha=config["alpha"])
print(pi_star)


# %% [code]
def expected_return(Q, p, r, alpha=0.01, rho0=None):
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

J = expected_return(Q_star, p_true, r_true, alpha=config["alpha"])
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

n_states = p_true.shape[0]
n_actions = p_true.shape[1]
theta_dim = n_states * n_actions + n_states * n_actions * n_states
theta = np.random.randn(theta_dim)
kappa = config["kappa"]
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
    alpha = 0.01

    for s in range(n_states):
        for a in range(n_actions):
            B_true_Q = 0.0
            for sp in range(n_states):
                logsumexp_sp = stable_logsumexp_alpha(Q_star[sp], alpha)

                B_true_Q += p_true[s, a, sp] * logsumexp_sp

            loss += (B_true_Q - Q_star[s, a]) ** 2
    return loss


loss = omd_loss(theta, kappa, n_states, n_actions, p_true, r_true)
print(loss)


# %% [code]
def optimize(loss_fn, theta_init, kappa, n_states, n_actions, p_true, r_true, learning_rate=0.1, inner_loop_steps=100, grad_step_size = 1e-5,  max_iterations=1000, print_every=20):

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

        if print_every and outer_loop % print_every == 0:
            print(f"iter {outer_loop}: loss = {losses[-1]:.4f}, ‖θ‖ = {np.linalg.norm(theta):.4f}")

        if np.linalg.norm(theta) > kappa:
            theta = (kappa / np.linalg.norm(theta)) * theta

    print(f"final:   loss = {losses[-1]:.4f}, ‖θ‖ = {np.linalg.norm(theta):.4f}")
    return theta, losses


 # MLE
theta_init = np.random.randn(theta_dim)
theta_mle, losses_mle = optimize(
   mle_loss,
   theta_init,
   kappa=config["kappa"],
   n_states=n_states,
   n_actions=n_actions,
   p_true=p_true,
   r_true=r_true,
   learning_rate=config["mle_learning_rate"],
   grad_step_size=config["fd_step"],
   max_iterations=config["mle_max_iterations"],
   print_every=config["mle_print_every"],
)

print(f"MLE final loss: {losses_mle[-1]:.4f}")

# OMD
theta_init = np.random.randn(theta_dim)
theta_omd, losses_omd = optimize(
    omd_loss,
    theta_init,
    kappa=config["kappa"],
    n_states=n_states,
    n_actions=n_actions,
    p_true=p_true,
    r_true=r_true,
    learning_rate=config["mle_learning_rate"],
    grad_step_size=config["fd_step"],
    max_iterations=config["mle_max_iterations"],
    print_every=config["mle_print_every"],
)

print(f"OMD final loss: {losses_omd[-1]:.4f}")


# %% [code]
def model_bellman_loss(Q, theta, kappa, n_states=None, n_actions=None, gamma=0.9, alpha=0.01):
    if n_states is None:
        n_states = Q.shape[0]
    if n_actions is None:
        n_actions = Q.shape[1]
    r_theta, p_theta = theta_to_model(theta, n_states, n_actions, kappa)
    B_theta_Q = soft_bellman_opt_explicit(Q, p_theta, r_theta, gamma, alpha)

    loss = 0.0
    for s in range(n_states):
        for a in range(n_actions):
            loss += (Q[s, a] - B_theta_Q[s, a]) ** 2
    return loss


r_theta, p_theta = theta_to_model(theta, n_states, n_actions, kappa)
B_theta_Q = soft_bellman_opt_explicit(Q, p_theta, r_theta)

print("theta:\n", theta)
print("\nQ:\n", Q)
print("\nB_theta_Q:\n", B_theta_Q)
print("\nLoss:", model_bellman_loss(Q, theta, kappa, n_states, n_actions))


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
            theta_init = np.random.randn(theta_dim)

            # MLE
            theta_mle, _ = optimize(
                mle_loss,
                theta_init,
                kappa,
                n_states,
                n_actions,
                p_true,
                r_true,
                learning_rate=config["mle_learning_rate"],
                grad_step_size=config["fd_step"],
                max_iterations=config["mle_max_iterations"],
                print_every=None,
            )
            r_m, p_m = theta_to_model(theta_mle, n_states, n_actions, kappa)
            Q_m, _ = value_iteration(p_m, r_m)
            J_m = expected_return(Q_m, p_true, r_true)
            results["mle"][kappa].append(J_m)

            # OMD (same init as MLE)
            theta_omd, _ = optimize(
                omd_loss,
                theta_init,
                kappa,
                n_states,
                n_actions,
                p_true,
                r_true,
                learning_rate=config["mle_learning_rate"],
                grad_step_size=config["fd_step"],
                max_iterations=config["mle_max_iterations"],
                print_every=None,
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


# %% [code]
def optimize_Q(loss_fn, theta, Q_init, kappa,
               learning_rate=0.01, max_iterations=100, grad_step_size=1e-5, print_every=None):

    Q = Q_init.copy()
    n_states, n_actions = Q.shape
    losses = []

    for step in range(max_iterations):
        grad = np.zeros_like(Q)
        for s in range(n_states):
            for a in range(n_actions):
                # two points allow for more accurate without going crazy - it's a slope
                # going from one point to two is the biggest value add
                Q_minus = Q.copy()
                Q_plus = Q.copy()

                Q_minus[s, a] -= grad_step_size
                Q_plus[s, a] += grad_step_size

                loss_minus = loss_fn(Q_minus, theta, kappa, n_states, n_actions)
                loss_plus = loss_fn(Q_plus, theta, kappa, n_states, n_actions)

                grad[s, a] = (loss_plus - loss_minus) / (2 * grad_step_size)

        Q -= learning_rate * grad

        loss = loss_fn(Q, theta, kappa, n_states, n_actions)
        losses.append(loss)

        if print_every and step % print_every == 0:
            print(f"Q step {step}: loss = {loss:.4f}")

    if print_every:
        print(f"Q Final: loss = {losses[-1]:.4f}")

    return Q, losses


# Test optimize_Q
Q_init = np.zeros((n_states, n_actions))
Q_trained, Q_losses = optimize_Q(
    model_bellman_loss,
    theta,
    Q_init,
    kappa,
    learning_rate=config["inner_q_learning_rate"],
    max_iterations=config["inner_q_max_iterations"],
    grad_step_size=config["fd_step"],
    print_every=config["inner_q_print_every"],
)

print("\nTrained Q:")
print(Q_trained)
print("\nFinal model Bellman loss:", model_bellman_loss(Q_trained, theta, kappa, n_states, n_actions))


# %% [code]
def true_bellman_loss_Q(Q, p_true, r_true, gamma=0.9, alpha=0.01):
    n_states = Q.shape[0]
    n_actions = Q.shape[1]

    B_theta_Q = soft_bellman_opt_explicit(Q, p_true, r_true, gamma, alpha)

    loss = 0.0
    for s in range(n_states):
        for a in range(n_actions):
            loss += (Q[s, a] - B_theta_Q[s, a]) ** 2
    return loss

true_loss = true_bellman_loss_Q(Q_trained, p_true, r_true)
print("True Bellman loss:", true_loss)
print("Return:", expected_return(Q_trained, p_true, r_true))


def implicit_gradient(theta, kappa, p_true, r_true, grad_step_size=1e-5, gamma=0.9, alpha=0.01):
    n_states = p_true.shape[0]
    n_actions = p_true.shape[1]

    # inner fixed point: Q*(theta)
    r_theta, p_theta = theta_to_model(theta, n_states, n_actions, kappa)
    Q_star, _ = value_iteration(p_theta, r_theta, gamma=gamma, alpha=alpha)

    Q_flat = Q_star.flatten()
    n_q = len(Q_flat)
    n_theta = len(theta)

    # f(theta, Q) = Q - B_theta(Q)
    def f_residual(Q_flat_local, theta_local):
        Q_local = Q_flat_local.reshape((n_states, n_actions))
        r_local, p_local = theta_to_model(theta_local, n_states, n_actions, kappa)
        B_local = soft_bellman_opt_explicit(Q_local, p_local, r_local, gamma=gamma, alpha=alpha)
        return (Q_local - B_local).flatten()

    # Jacobian df/dQ [n_q, n_q]
    df_dQ = np.zeros((n_q, n_q))
    for j in range(n_q):
        Q_plus = Q_flat.copy()
        Q_minus = Q_flat.copy()
        Q_plus[j] += grad_step_size
        Q_minus[j] -= grad_step_size
        df_dQ[:, j] = (f_residual(Q_plus, theta) - f_residual(Q_minus, theta)) / (2 * grad_step_size)

    # Jacobian df/dtheta [n_q, n_theta]
    df_dtheta = np.zeros((n_q, n_theta))
    for j in range(n_theta):
        theta_plus = theta.copy()
        theta_minus = theta.copy()
        theta_plus[j] += grad_step_size
        theta_minus[j] -= grad_step_size
        df_dtheta[:, j] = (f_residual(Q_flat, theta_plus) - f_residual(Q_flat, theta_minus)) / (2 * grad_step_size)

    # gradient dJ/dQ under true MDP
    dJ_dQ = np.zeros(n_q)
    for j in range(n_q):
        Q_plus = Q_flat.copy()
        Q_minus = Q_flat.copy()
        Q_plus[j] += grad_step_size
        Q_minus[j] -= grad_step_size

        J_plus = expected_return(Q_plus.reshape((n_states, n_actions)), p_true, r_true, alpha=alpha)
        J_minus = expected_return(Q_minus.reshape((n_states, n_actions)), p_true, r_true, alpha=alpha)

        dJ_dQ[j] = (J_plus - J_minus) / (2 * grad_step_size)

    # IFT: dQ/dtheta = -(df/dQ)^(-1) (df/dtheta)
    # chain: dJ/dtheta = (dJ/dQ) (dQ/dtheta)
    damping = 1e-8
    system = df_dQ + damping * np.eye(n_q)
    dQ_dtheta = -np.linalg.solve(system, df_dtheta)
    dJ_dtheta = dJ_dQ @ dQ_dtheta

    J_current = expected_return(Q_star, p_true, r_true, alpha=alpha)
    return dJ_dtheta, Q_star, J_current


def optimize_outer(theta_init, kappa, p_true, r_true,
                   learning_rate=0.01, max_iterations=1000,
                   grad_step_size=1e-5, print_every=100):
    theta = theta_init.copy()
    returns = []

    for step in range(max_iterations):
        grad, _, _ = implicit_gradient(
            theta, kappa, p_true, r_true, grad_step_size=grad_step_size
        )

        # maximize J
        theta += learning_rate * grad

        # project to ||theta|| <= kappa
        if np.linalg.norm(theta) > kappa:
            theta = (kappa / np.linalg.norm(theta)) * theta

        n_states = p_true.shape[0]
        n_actions = p_true.shape[1]
        r_theta, p_theta = theta_to_model(theta, n_states, n_actions, kappa)
        Q_star, _ = value_iteration(p_theta, r_theta)
        J = expected_return(Q_star, p_true, r_true)
        returns.append(J)

        if print_every and step % print_every == 0:
            print(f"Outer step {step}: J = {J:.4f}, ‖θ‖ = {np.linalg.norm(theta):.4f}")

    if print_every:
        print(f"Outer Final: J = {returns[-1]:.4f}, ‖θ‖ = {np.linalg.norm(theta):.4f}")

    return theta, returns


def find_convergence_point(J_history, window, eps):
    """
    Returns the first step at which J has changed by less than eps
    for `window` consecutive steps. Falls back to the final step.
    """
    diffs = np.abs(np.diff(J_history))
    for i in range(window, len(diffs)):
        if np.all(diffs[i - window:i] < eps):
            return i
    return len(J_history) - 1


def collect_calibration_transitions(theta_final, Q_star, kappa,
                                    n_states, n_actions, n_transitions,
                                    gamma=0.9, alpha=0.01, rng=None):
    """
    Freeze theta. Sample actions from the converged policy softmax(Q*(s)/alpha)
    and next states from the learned model p_theta(s'|s,a).
    """
    if rng is None:
        rng = np.random.default_rng()

    r_theta, p_theta = theta_to_model(theta_final, n_states, n_actions, kappa)

    transitions = []
    s = rng.integers(0, n_states)

    for _ in range(n_transitions):
        q_vals = Q_star[s] / alpha
        q_vals = q_vals - np.max(q_vals)
        probs = np.exp(q_vals) / np.sum(np.exp(q_vals))
        a = rng.choice(n_actions, p=probs)

        sp = rng.choice(n_states, p=p_theta[s, a])

        transitions.append((s, a, sp))
        s = sp

    return transitions, r_theta, p_theta


def compute_calibration_residuals(transitions, Q_star, p_true, r_true,
                                  gamma=0.9, alpha=0.01):
    """
    Nonconformity score: Q*(s,a) - B^true(Q*)(s,a)
    """
    B_Q = soft_bellman_opt_explicit(Q_star, p_true, r_true, gamma=gamma, alpha=alpha)
    residuals = []
    for (s, a, sp) in transitions:
        residuals.append(Q_star[s, a] - B_Q[s, a])
    return np.array(residuals)


def compute_q_intervals(residuals, Q_star, alpha=0.1):
    """
    Standard split conformal quantile over scalar residuals.
    """
    q_lo = np.quantile(residuals, alpha / 2)
    q_hi = np.quantile(residuals, 1 - alpha / 2)

    return {
        "lower": Q_star - q_hi,
        "upper": Q_star - q_lo,
        "width": q_hi - q_lo,
        "q_lo": q_lo,
        "q_hi": q_hi,
    }


# OMD experiment (IFT gradient)
print("\n=== OMD (IFT) ===")
theta_init = np.random.randn(theta_dim)
theta_omd, returns_omd = optimize_outer(
    theta_init,
    kappa,
    p_true,
    r_true,
    learning_rate=config["outer_learning_rate"],
    max_iterations=config["outer_max_iterations"],
    grad_step_size=config["outer_grad_step"],
    print_every=config["outer_print_every"],
)

# Evaluate final Q from converged model via fixed point iteration
r_omd, p_omd = theta_to_model(theta_omd, n_states, n_actions, kappa)
Q_final, _ = value_iteration(p_omd, r_omd)

print("\nFinal Q:")
print(Q_final)
print("True Bellman loss:", true_bellman_loss_Q(Q_final, p_true, r_true))
print("Return:", expected_return(Q_final, p_true, r_true))

# Direct MLE vs OMD-IFT return comparison at same kappa
r_mle, p_mle = theta_to_model(theta_mle, n_states, n_actions, config["kappa"])
Q_mle, _ = value_iteration(
    p_mle,
    r_mle,
    gamma=config["gamma"],
    alpha=config["alpha"],
    tol=config["value_iteration_tol"],
    max_iter=config["value_iteration_max_iter"],
)
J_mle = expected_return(Q_mle, p_true, r_true, alpha=config["alpha"])

print("MLE Return:", J_mle)
print("OMD-IFT Return:", returns_omd[-1])

# Calibration phase
converged_at = find_convergence_point(
    returns_omd,
    window=config["convergence_window"],
    eps=config["convergence_eps"],
)

transitions, r_theta_cal, p_theta_cal = collect_calibration_transitions(
    theta_omd,
    Q_final,
    kappa,
    n_states,
    n_actions,
    n_transitions=config["calibration_n_transitions"],
    gamma=config["gamma"],
    alpha=config["alpha"],
    rng=np.random.default_rng(config["seed"] + 1),
)

residuals = compute_calibration_residuals(
    transitions,
    Q_final,
    p_true,
    r_true,
    gamma=config["gamma"],
    alpha=config["alpha"],
)

q_intervals = compute_q_intervals(
    residuals,
    Q_final,
    alpha=config["calibration_alpha"],
)

state_labels = [f"s{s}" for s in range(n_states)]
action_labels = [f"a{a}" for a in range(n_actions)]

print("\n=== CALIBRATION DIAGNOSTICS ===")
print(f"Converged at step: {converged_at} / {config['outer_max_iterations']}")
print(f"Calibration transitions: {len(transitions)}")
print(f"Residual mean: {np.mean(residuals):.4f}")
print(f"Residual std:  {np.std(residuals):.4f}")
print(f"Interval width (uniform): {q_intervals['width']:.4f}")
print(f"q_lo: {q_intervals['q_lo']:.4f}  q_hi: {q_intervals['q_hi']:.4f}")

print("\n=== Q INTERVALS ===")
for s in range(n_states):
    for a in range(n_actions):
        lo = q_intervals["lower"][s, a]
        hi = q_intervals["upper"][s, a]
        q = Q_final[s, a]
        print(f"  {state_labels[s]},{action_labels[a]}: [{lo:.3f}, {hi:.3f}]  (Q*={q:.3f})")

if converged_at == config["outer_max_iterations"] - 1:
    print("WARNING: convergence fallback — consider increasing outer_max_iterations")
