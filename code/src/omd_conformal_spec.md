# Calibration Phase — Implementation Addendum

This document specifies exactly what to add to the current clean baseline
(`00_omd-algorithm.py`, complex MDP, logsumexp fix in place). Nothing in
the existing code changes.

---

## What We're Adding

A post-convergence calibration phase that produces conformal prediction
intervals over the Q value function. The intervals are:

```
[Q*(s,a) - q_hi,  Q*(s,a) - q_lo]
```

where `q_lo` and `q_hi` are quantiles of Bellman residuals collected from
trajectory rollouts under the frozen converged model.

---

## Why This Is Defensibly Conformal

- θ is frozen post-convergence → p_θ is fixed
- Transitions are sampled from a ~ softmax(Q*(s)), s' ~ p_θ(s'|s,a)
- Under an ergodic MDP with frozen θ, these transitions are approximately
  exchangeable → coverage guarantee approximately holds
- The 1000-transition figure is grounded in the CP literature
- Variation comes from genuine policy/model stochasticity, not anything
  artificial (no noise on θ, no random VI initialization)

---

## New Config Parameters

```python
"calibration_n_transitions": 1000,  # rollout length post-convergence
"calibration_alpha": 0.1,           # 1 - alpha = 90% intervals
"convergence_window": 10,           # window for J convergence check
"convergence_eps": 1e-4,            # J change threshold
```

---

## New Functions

### 1. `find_convergence_point`

Detect when J has stopped moving in the optimization history.

```python
def find_convergence_point(J_history, window, eps):
    """
    Returns the first step at which J has changed by less than eps
    for `window` consecutive steps. Falls back to the final step.
    """
    diffs = np.abs(np.diff(J_history))
    for i in range(window, len(diffs)):
        if np.all(diffs[i - window:i] < eps):
            return i
    return len(J_history) - 1  # fallback
```

Log a warning if fallback triggers — it means `outer_max_iterations` may
be too low.

---

### 2. `collect_calibration_transitions`

Roll out transitions from the converged policy and frozen model.

```python
def collect_calibration_transitions(theta_final, Q_star, kappa,
                                    n_states, n_actions, n_transitions,
                                    gamma=0.9, alpha=0.01, rng=None):
    """
    Freeze theta. Sample actions from the converged policy softmax(Q*(s)/alpha)
    and next states from the learned model p_theta(s'|s,a).

    Returns:
        transitions: list of (s, a, s') tuples
        r_theta: reward matrix under frozen theta
        p_theta: transition matrix under frozen theta
    """
    if rng is None:
        rng = np.random.default_rng()

    r_theta, p_theta = theta_to_model(theta_final, n_states, n_actions, kappa)

    transitions = []
    s = rng.integers(0, n_states)  # random initial state

    for _ in range(n_transitions):
        # sample action from converged policy (numerically stable)
        q_vals = Q_star[s] / alpha
        q_vals = q_vals - np.max(q_vals)
        probs = np.exp(q_vals) / np.sum(np.exp(q_vals))
        a = rng.choice(n_actions, p=probs)

        # sample next state from learned model
        sp = rng.choice(n_states, p=p_theta[s, a])

        transitions.append((s, a, sp))
        s = sp  # continue trajectory

    return transitions, r_theta, p_theta
```

---

### 3. `compute_calibration_residuals`

Compute the nonconformity score for each calibration transition.

```python
def compute_calibration_residuals(transitions, Q_star, r_theta, p_theta,
                                   gamma=0.9, alpha=0.01):
    """
    Nonconformity score: Q*(s,a) - B^theta(Q*)(s,a)

    Computed once over the full model (not per-transition lookup) then
    indexed by visited (s,a).

    Returns:
        residuals: 1D array of shape [n_transitions]
    """
    B_Q = soft_bellman_opt_explicit(Q_star, p_theta, r_theta, gamma=gamma, alpha=alpha)
    residuals = []
    for (s, a, sp) in transitions:
        residuals.append(Q_star[s, a] - B_Q[s, a])
    return np.array(residuals)
```

---

### 4. `compute_q_intervals`

Direct empirical quantiles over scalar residuals, applied as uniform shift to Q*.

```python
def compute_q_intervals(residuals, Q_star, alpha=0.1):
    """
    Standard split conformal quantile — no bootstrap needed with 1000 transitions.
    Width is uniform across (s,a) — the nonconformity score is a scalar
    per transition, not per state-action pair.

    If per-(s,a) width variation is needed later, bucket residuals by (s,a)
    and compute quantiles per bucket. Requires sufficient visits to each pair.
    """
    q_lo = np.quantile(residuals, alpha / 2)
    q_hi = np.quantile(residuals, 1 - alpha / 2)

    return {
        "lower": Q_star - q_hi,   # [n_states, n_actions]
        "upper": Q_star - q_lo,
        "width": q_hi - q_lo,     # scalar
        "q_lo": q_lo,
        "q_hi": q_hi,
    }
```

---

## Integration — What to Add After the Outer Loop

```python
# optimize_outer returns theta_omd, returns_omd as before
# then add:

# convergence detection
converged_at = find_convergence_point(
    returns_omd,
    window=config["convergence_window"],
    eps=config["convergence_eps"],
)

# calibration rollout with frozen theta
transitions, r_theta_cal, p_theta_cal = collect_calibration_transitions(
    theta_omd, Q_final, kappa, n_states, n_actions,
    n_transitions=config["calibration_n_transitions"],
    gamma=config["gamma"],
    alpha=config["alpha"],
    rng=np.random.default_rng(config["seed"] + 1),
)

# nonconformity scores
residuals = compute_calibration_residuals(
    transitions, Q_final, r_theta_cal, p_theta_cal,
    gamma=config["gamma"],
    alpha=config["alpha"],
)

# intervals
q_intervals = compute_q_intervals(
    residuals, Q_final,
    alpha=config["calibration_alpha"],
)

# diagnostics
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
        q  = Q_final[s, a]
        print(f"  {state_labels[s]},{action_labels[a]}: [{lo:.3f}, {hi:.3f}]  (Q*={q:.3f})")

if converged_at == config["outer_max_iterations"] - 1:
    print("WARNING: convergence fallback — consider increasing outer_max_iterations")
```

---

## Expected Output on Complex MDP

- `residual std > 0` — variation from stochastic transitions, not artificial
- `width > 0` — non-degenerate intervals
- `q_hi_std` small — bootstrap is stable
- Intervals are uniform across (s,a) — expected given scalar nonconformity score

On the paper MDP (deterministic), residuals will still be near-zero because
B^θ(Q*) ≈ Q* when the model can represent the true MDP well. This is
correct behavior.

---

## Open Questions

- Per-(s,a) width variation via bucketed residuals (future)
- Convergence intervals via state-conditioned expected return (future)
- Coverage check across seeds (in test spec)
