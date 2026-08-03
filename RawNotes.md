
## MLE

- General MLE = adjust θ to maximize p(data|θ).
- All mean-based (MSE) are MLE (under Gaussian assumption)
- Not all MLE are mean-based (could be median, mode, quantiles, etc.)
- Even with perfect likelihood you can have bad "control" ie decisions
    - Go left, go right average to straight ahead which ma be a wall
    - An outcome that occurs 99 percent of the time so the model just says "yes" every time
- In OMD we optimize **expected return directly** instead of prediction accuracy.

## Bellman Optimality Operator

- For a given model θ = (p_θ, r_θ), the Bellman optimality operator B^θ takes any Q-function and returns a "better" Q-function
- This feels a little metaphysical, but what it means is a "one-step lookahead" operator".
    - If we act greedily with regards to Q for one step and keep doing that, what do we get (Q)?

### Notation interlude: Why the superscript θ? Because the Bellman operator depends on the model:
- It's not B(theta)=>Q, it's Qnew = model(theta).bellman(Qold)
- This is called a "contraction mapping" - the distance between two inputs shrinks after application of this operator (by a scaling factor γ < 1)
- The model will not converge if it's not strictly made to contract (or at least we can't prove that it will)


### Jacobian interlude:

- A simple derivative is a single input -> single output
    - Tells us:
        - The slope of a curve
        - The rate of change
- A gradient derivative is multiple inputs -> single output
    - This tells us:
        - The gradient of the steepest slope
        - Which input matters most
        - Optimal direction: moving the opposite of the gradient will minimize the function (which is a loss function)
- A Jacobian derivative is multiple inputs -> multiple outputs
    - This tells us:
        - How the entire output space stretches/rotates when input changes
        - Which input affects which output.
        - Linear approximation: F(x + Δx) ≈ F(x) + J · Δx
            - The best linear approximation of a non-linear thing

## Implicit Function Theorem

- The heart of this is that we want the derivative of something with no closed form solution.
- Instead of calculating the solution, we might observe how changes in the input affect the output (numerical differentiation).
    - This involves re-solving the problem for each input change.
    - IFT allows for more efficient estimation of what that would tell us.
        - It doesn't re-solve for Q* at each θ
        - It leverages the fact that F(θ, Q*) = 0 always holds
        - This gives us a formula for dQ*/dθ using only local derivatives ∂F/∂θ and ∂F/∂Q at the current point
    - This is done by linear approximation.
        - This is like drawing a polygon around an output space like N64 graphics for calculus.
    - It acts on the theta and Q in the loop, not the global outer ones - if it was easy to do we wouldn't need this algorithm.

## OMD Two-step pipeline

- Step 1: Learn model (p_θ, r_θ) using MLE (maximize likelihood of observed transitions)
- Step 2: Use learned model for planning (compute policy π that maximizes return)
    - Planning just means policy output

### Methods recap interlude
- Given: A model (p_θ, r_θ), compute policy π that maximizes expected return
    - Methods:
        - Value iteration: Iteratively apply Bellman operator to get Q*
        - Policy iteration: Alternate between policy evaluation and improvement
        - MPC (Model Predictive Control): Plan trajectory online, execute first action
        - Search (MCTS): Simulate trajectories using model, pick best action

## General notes on intro
- Error matters when it impacts decisions and that all, we can ignore irrelevant details
- Objective mismatch is the delta between MLE accuracy and policy optimality
    - Concrete example:
        - Good outcome: Robot vision prediction matches the next pixel
        - Good outcome: Robot avoids walking into a wall
        - Nitpick:
            - I originally said the following:
                - Robot gets reward for predicting the next pixel
                - Robot gets reward for not walking into a wall
                - It's worth noting there is no "reward" as such in MLE it's just the MLE
                - There is a reward as such in policy optimality

## Robust methods vs OMD
 - Robust/uncertainty methods:
    - Provide meaningful uncertainty estimates _around_ the MLE but still "optimize" based on it
    - Generally use this to make conservative/worst-case predictions

 - OMD:
    - This method ignores MLE entirely and trains the model to maximize return directly
    - No uncertainty provided — model is "good" if it makes agent perform well

## Soft Bellman and formulation
- The Soft Bellman is used because the hard one isn't differentiable

## Components

A reminder:
- Theta is the model params, a tuple, in the case it has p and r (of theta)
- Q is a function that estimates a return given s and a
- Q(s, a) is an application of Q that gives expected future return
    - Note "reward" vs "return" - one is an individual result, the other is the cumulative sum of all of rewards
- Q* is the optimal action value function which corresponds to a fixed point of the Q function space
    - Note: the is the difference between talking about a function `foo()` and using it `foo(x)`
- B^theta is the Bellman operator which estimates the return if you wake a given step and then follow the best known action for the rest of the run
- Pi of Q is the policy that maximizes expected return given a certain Q function ands state
    - This can be though of as the best rule to follow given a given state of the model
- J(pi) - expected return of the pi
- F(theta, Q) is Bellman residual which is the delta between the current Q and the optimal one we project with the Bellman operator

 ┌─────────┬───────────────────────┬───────────────────────────────────┐
 │ Symbol  │ Meaning               │ Input → Output                    │
 ├─────────┼───────────────────────┼───────────────────────────────────┤
 │ θ       │ Model parameters      │ (p_θ, r_θ) — transition + reward  │
 ├─────────┼───────────────────────┼───────────────────────────────────┤
 │ Q       │ Action-value function │ (s, a) → ℝ                        │
 ├─────────┼───────────────────────┼───────────────────────────────────┤
 │ Q*      │ Optimal Q-function    │ Fixed point of B^θ                │
 ├─────────┼───────────────────────┼───────────────────────────────────┤
 │ B^θ     │ Soft Bellman operator │ Q → Q' (one-step lookahead)       │
 ├─────────┼───────────────────────┼───────────────────────────────────┤
 │ π_Q     │ Policy from Q         │ s → distribution over A (softmax) │
 ├─────────┼───────────────────────┼───────────────────────────────────┤
 │ J(π)    │ Expected return       │ π → ℝ (scalar value)              │
 ├─────────┼───────────────────────┼───────────────────────────────────┤
 │ F(θ, Q) │ Bellman residual      │ (θ, Q) → 0 at fixed point         │
 └─────────┴───────────────────────┴───────────────────────────────────┘

 The Chain

 ```
   θ  →  Q*  →  π_Q*  →  J
   │      │       │      │
   │      │       │      └─ Expected return (what we want to max)
   │      │       └──────── Policy (softmax over Q*)
   │      └──────────────── Optimal Q (fixed point: Q* = B^θ Q*)
   └─────────────────────── Model params (we update these)
 ```

 The Problem

 - We want ∂J/∂θ
 - But Q = ϕ(θ)* has no closed form (it's "iterate B^θ until convergence")
 - Can't differentiate through infinite iterations


 ## OMD

Updated chart:
   ┌─────────┬───────────────────────┬───────────────────────────────────┐
   │ Symbol  │ Meaning               │ Input → Output                    │
   ├─────────┼───────────────────────┼───────────────────────────────────┤
   │ θ       │ Model parameters      │ (p_θ, r_θ) — transition + reward  │
   ├─────────┼───────────────────────┼───────────────────────────────────┤
   │ Q       │ Action-value function │ (s, a) → ℝ                        │
   ├─────────┼───────────────────────┼───────────────────────────────────┤
   │ Q*      │ Optimal Q-function    │ Fixed point of B^θ                │
   ├─────────┼───────────────────────┼───────────────────────────────────┤
   │ B^θ     │ Soft Bellman operator │ Q → Q' (one-step lookahead)       │
   ├─────────┼───────────────────────┼───────────────────────────────────┤
   │ π_Q     │ Policy from Q         │ s → distribution over A (softmax) │
   ├─────────┼───────────────────────┼───────────────────────────────────┤
   │ J(π)    │ Expected return       │ π → ℝ (scalar value)              │
   ├─────────┼───────────────────────┼───────────────────────────────────┤
   │ F(θ, Q) │ Bellman residual      │ (θ, Q) → 0 at fixed point         │
   ├─────────┼───────────────────────┼───────────────────────────────────┤
   │ κ       │ Norm bound            │ Controls capacity: {θ : ‖θ‖ ≤ κ}  │
   │         │                       │ Small κ → misspecified model      │
   └─────────┴───────────────────────┴───────────────────────────────────┘

Kappa is used to create model misspecification. It's not needed for the algorithm itself but it's a great diagnostic.



