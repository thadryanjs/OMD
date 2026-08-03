# 1-Month Mastery + Extension Plan (OMD | Conformal OPE | ILP)

**Goal:** Deep understanding + extension ideas by month-end  
**Audience:** Applied stats background, strong coding, ramp-up on theory  
**Output:** Working notes, proofs, code sketches, 3–5 mix ideas

---

## Week 1: Foundations + Cross-Paper Map (5–6 hrs/wk)

**Objective:** Orient on problem domains, identify connections early.

### Monday–Tuesday: OMD paper skim + core insight
- Skim Sections 1–3 (problem, preliminaries)
- Core: MLE ≠ control objective (objective mismatch)
- Key insight: Treat Q* as implicit fn of θ → differentiate via IFT
- Note: Soft Bellman operator (log-sum-exp) for differentiability
- **To understand:** Why soft Bellman? (MaxEnt RL connection)

### Wednesday: Conformal OPE paper skim + core insight
- Skim Sections I–III (intro, related, preliminaries)
- Core: Off-policy eval needs non-asymptotic coverage guarantees
- Key insight: Weighted exchangeability handles π_b → π shift
- Note: Asymmetric/double-quantile scores reduce interval length
- **To understand:** Why weighted CP beats importance sampling alone?

### Thursday: ILP paper skim + core insight
- Skim Sections 1–2 (intro, preliminaries)
- Core: Uncertain params (intervals) → exact optimal solution sets
- Key insight: B-stability + basis conditions determine solution zones
- Note: ISOM-2 ensures both feasible AND optimal
- **To understand:** Why basis stability is structural anchor?

### Friday: Cross-paper worksheet
- **Matrix:** For each paper, list: (problem class | uncertainty type | solution concept | robustness mechanism)
- **Initial mix ideas:** (record 2–3 rough hunches)
  - e.g., "Can conformal bounds wrap OMD model predictions?"
  - e.g., "Do interval constraints tighten OPE evaluation bounds?"
- **Shared patterns:** gradient-based optimization, distribution shift, feasibility→optimality

**Deliverable:** Cross-paper map + hunch list

---

## Week 2: Deep OMD (8–10 hrs/wk)

**Objective:** Master IFT, implicit differentiation, model misspec.

### Monday–Tuesday: Implicit function theorem + IFT in OMD
- Read Section 4.1 carefully (Theorem 1, Eq. 5, 7)
- **Drill:** Prove Jacobian inversion formula in Eq. 7 for 2×2 example
- **Code sketch:** Implement IFT backward pass (JAX implicit diff or torch.autograd.Function)
- **Intuition:** Why only need final fixed point w*, not trajectory of iterations?

### Wednesday: Model misspec + approximation bounds
- Read Section 4.2 (Figure 1, bounded-norm setting)
- Read Section 5.2 (Theorem 2: Q* approximation error)
- **Compare:** OMD bound vs MLE bound (Eq. in Theorem 2)
- **Key:** OMD error scales with Bellman mismatch ε; MLE with policy divergence × model error
- **Derive:** Why OMD tighter under capacity constraint?
- **Code:** Reproduce Figure 1 experiment (2-state MDP)

### Thursday: Function approximation + practical OMD
- Read Section 6 (Q-network, first-order optimality, bi-level optimization)
- **Understand:** Why inner loop trains Q; outer loop trains θ?
- **Critical detail:** Jacobian approximation (identity matrix!) — why works despite theory?
- **Code:** Build OMD Algorithm 1 skeleton (inner/outer loop structure)

### Friday: Synthesis + extension brainstorm
- **Write:** 1-page proof sketch of Theorem 2 in plain language
- **Experiment idea:** What if Q is conformal (has interval estimates)? How propagate to θ gradient?
- **Code:** Add option to OMD for Q-function uncertainty (e.g., dropout/ensemble)
- **Question:** Can we prove OMD converges under distribution shift (different behavior policy)?

**Deliverable:** Theorem 2 proof + OMD Algorithm 1 code + 1 extension idea doc

---

## Week 3: Deep Conformal OPE (8–10 hrs/wk)

**Objective:** Master weighted CP, distribution shift, score design.

### Monday–Tuesday: Standard CP + weighted exchangeability
- Read Section III.B (split CP, Eq. 1)
- **Intuition:** Why quantile construction ensures coverage? (exchange symmetry)
- Read Section IV.A (weighted CP, Proposition 1 + proof)
- **Key:** Weighted exchangeability + likelihood ratio w(x,y) = dP^π/dP^π_b
- **Code:** Implement split CP from scratch on toy regression problem

### Wednesday: Score functions + asymmetry
- Read Section IV.B.1 (double-quantile, Eq. 7, Proposition 3)
- **Intuition:** Why symmetric pinball score centers on π_b? Why break symmetry?
- **Visualize:** Figure 2 — sketch confidence set for shifted policies
- **Code:** Implement pinball score vs double-quantile on 1D problem

### Thursday: Likelihood ratio estimation + optimality
- Read Section V (Monte-Carlo, empirical, gradient methods)
- **Critical:** Why IS has high variance for long horizons? (exponential importance weights)
- **Proposition 5:** Concentration bounds on weight estimation
- **Code:** Implement empirical weight estimator (Eq. 13) for tabular case
- **Question:** Can ILP uncertainty bounds help with weight estimation robustness?

### Friday: Synthesis + extension brainstorm
- **Write:** 1-page on why weighted CP beats standard CI methods (sketch Prop. 2 vs standard bounds)
- **Experiment idea:** If OMD model is uncertain (interval predictions), how does that propagate to OPE bounds?
- **Code:** Sketch conformal OPE that accepts ensemble/interval model predictions
- **Question:** Can we use ILP-style constraint uncertainty for π_b vs π difference?

**Deliverable:** Weighted CP code + Prop. 5 concentration inequality derivation + 1 extension idea doc

---

## Week 4: ILP Intuition + Ideation (10–12 hrs/wk)

**Objective:** Master basis stability + ILP solution set concepts. Synthesize + propose extensions.

### Monday: ILP fundamentals (fast track)
- Read Sections 2.1–2.2 (interval LP formulation, B-stability, Theorems 1–6)
- **Key:** Extreme zones, basis stability, optimal solution SET (not point)
- **Intuition:** Why interval LP needed? (uncertain params in real systems)
- **Code:** Implement interval arithmetic basics; test on 2-var example

### Tuesday: Existing methods survey (comparative lens)
- Read Sections 3.1–3.4 (BWC, TSM, MILP, ITSM) — track pros/cons
- **Focus:** What's infeasible? What's non-optimal? Why?
- **Figure 1:** Visualize feasible zones and why certain methods fail
- **Pattern:** All methods trade off extra constraints ↔ redundancy

### Wednesday: ISOM-2 deep dive + extensions
- Read Section 4 (ISOM-2, Theorems 9–10, Corollary 1)
- **Understand:** How does sign-reversal in Eq. 20 guarantee optimality?
- **Theorem 10 proof idea:** Active constraints determine which inequalities flip
- **Code:** Implement ISOM-2 for small numerical example (Eq. 24–25)

### Thursday–Friday: Cross-paper synthesis + ideation sprint

**Synthesis:**
- **Table:** (OMD | Conformal OPE | ILP) with columns: (Uncertainty type | Solution concept | Robustness mechanism | Math tool)
- **Observation:** OMD has model misspec (continuous); OPE has policy shift (weighted); ILP has param uncertainty (intervals)
- **Connection:** All three optimize under adversarial/worst-case constraints

**Ideation (generate 5–7 extension ideas, rank by feasibility × novelty):**

#### 1. **Conformal OMD Value Bounds** [feasibility: high | novelty: medium]
- Idea: Wrap OMD's value fn estimates in conformal prediction intervals
- Why: Quantify remaining uncertainty after model learning
- Math hook: IFT through conformal score function?
- Code effort: Medium (combine OMD + OPE)
- First step: Can we backprop through CP quantile operation?

#### 2. **Interval-Robust OPE Weights** [feasibility: high | novelty: medium]
- Idea: Model π_b, π as interval-uncertain policies; propagate to w(x,y) intervals
- Why: Robust importance sampling under policy misspecification
- Math hook: ILP interval arithmetic in likelihood ratio computation
- Code effort: Medium (interval propagation in OPE)
- First step: Formulate likelihood ratio as ILP constraint?

#### 3. **OMD with Feasibility Constraints (ILP-style)** [feasibility: medium | novelty: high]
- Idea: Solve OMD's model learning with uncertain reward/transition constraints (intervals)
- Why: Real-world systems have bounded uncertain parameters
- Math hook: Add ILP-style basis stability to OMD's bi-level optimization
- Code effort: Hard (modify OMD to handle interval params)
- First step: What does B-stability mean for neural net models?

#### 4. **Conformal Prediction for Solution Sets** [feasibility: medium | novelty: high]
- Idea: Use CP to bound the optimal solution set of OMD model (like ILP does)
- Why: Uncertain models → uncertain optimal solutions
- Math hook: CP on MDP solution space instead of scalars
- Code effort: Hard (new theory needed)
- First step: Is solution set property exchangeable under distribution shift?

#### 5. **Worst-Case OPE under Model Misspec** [feasibility: medium | novelty: high]
- Idea: Conformal OPE + OMD's misspec bound: tightest worst-case policy value?
- Why: Combines uncertainty quantification (OPE) + model quality (OMD)
- Math hook: Compose OMD Theorem 2 bound with OPE weighted CP bound
- Code effort: Medium-Hard
- First step: Can bounds be composed analytically or need empirical union?

#### 6. **ILP Formulation of OPE Feasibility** [feasibility: low | novelty: high]
- Idea: Frame OPE interval bounds as ILP optimal solution set problem
- Why: Might unify theoretical foundations of these three papers
- Math hook: Value interval [V-, V+] ~ optimal solution set in ILP
- Code effort: Very hard (requires new theory)
- First step: Can likelihood ratio weights be modeled as ILP parameters?

#### 7. **Adaptive OMD under Conformal Uncertainty** [feasibility: high | novelty: medium]
- Idea: Use conformal OPE predictions to guide which model parameters θ to update
- Why: Prioritize learning uncertain aspects of model
- Math hook: Gradient weighting by OPE confidence interval width
- Code effort: Medium
- First step: How to map OPE uncertainty to parameter importance?

**Ranking (1-month project feasibility):**
- **Quick wins (2–3 wks to working code):** #1, #2, #7
- **Medium depth (3–4 wks to working code):** #3, #5
- **Research-heavy (4+ wks):** #4, #6

**Recommend for 1-month goal:** Start with #1 or #7 (high leverage, medium code effort), outline #3 or #5 in parallel.

**Deliverable:** Ranked ideation doc + 1-page technical sketch of preferred extension (math + pseudocode)

---

## Parallel Throughout: Code & Experiment Log
- **Week 1:** Reproduce Figure 1 (OMD misspec), Figure 4 (OPE inventory), numerical example (ILP)
- **Week 2–3:** Incrementally implement OMD Algorithm 1, Conformal OPE Algorithm 1
- **Week 4:** Sketch prototype for chosen extension
- **Keep:** Working Python notebook with proofs, derivations, plots

---

## End-of-Month Deliverables

1. **Understanding doc:** 5–10 page math summary of each paper (theorem statements + intuition)
2. **Code repo:** Working implementations of OMD, Conformal OPE, ILP solver (can be partial)
3. **Extension proposal:** 2–3 page technical doc with math, pseudocode, preliminary results (or roadmap)
4. **Proof sketches:** 1-page plain-language derivations of key theorems (Nikishin Thm 2, Foffano Prop 1, Nehi Thm 10)
5. **Experiment notebook:** Reproducing paper figures + extended scenarios

---

## Quick Ref: Paper Connections

| Aspect          | OMD           | Conformal OPE        | ILP              |
|-----------------|---------------|----------------------|------------------|
| Problem         | Model → Return | Policy value ∈ [?,?] | Optimize ∈ [?,?] |
| Uncertainty     | Misspec       | Policy shift + model | Param intervals  |
| Solution obj    | θ (model)     | [V-, V+] interval    | x∈ solution set  |
| Robustness      | IFT gradient  | Weighted exchangeab  | Basis stability  |
| Math tool       | Implicit diff | Conformal pred       | Interval arith   |
| **Bridge idea** | Conformal bounds on θ gradient? | Interval policies → robust OPE? | CP on solution sets? |
