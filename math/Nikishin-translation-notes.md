# Nikishin et al. 2022 — Translation Notes

## Source materials
- **PDF:** `Nikishin_et_al__-_2022_-_Control-Oriented_Model-Based_Reinforcement_Learning_with_Implicit_Differentiation.pdf` (9 pages, AAAI-22 camera-ready)
- **PNG screenshots:** `Nikishin-1` through `Nikishin-9`, one per page, used to cross-check math against the embedded PDF text layer.

## Method
Both the embedded PDF text and the rendered page screenshots were read end-to-end. Every numbered equation and every algorithm line was checked against the screenshot before being committed to the markdown. The PDF text layer for this paper turned out to be quite clean — most discrepancies were minor encoding artifacts rather than real OCR failures.

---

## Discrepancies between raw PDF text and the screenshots

### 1. Vertical-bar / norm characters

The PDF text layer renders LaTeX `\|...\|` inside `$$...$$` as `k...k` (lowercase k as a stand-in for the double-bar). Examples (raw PDF text → corrected):

- `kfθ(s, a) − s'k²` → $\|f_\theta(s, a) - s'\|^2$ (eq. 1)
- `kθk ≤ κ` → $\|\theta\| \le \kappa$ (Section 4.2 + eq. 8)
- `kp(·|s, a) − pˆ(·|s, a)k₁` → $\|p(\cdot|s,a) - \hat p(\cdot|s,a)\|_1$ (Theorem 2)

Single-bar absolute value `|...|` survived correctly.

### 2. Latin extended characters in references

- `Schafer` ¨ → `Schäfer` (Bacon et al. 2019 reference; PDF puts the umlaut on a separate line as a combining diaeresis).
- `Kuttler` ¨ → `Küttler` (Beattie et al. 2016)
- `Vald es` → `Valdés` (Beattie et al. 2016)
- `Universite ´` `de Montr ´eal` → `Université de Montréal` (author affiliations)

These were corrected in the markdown.

### 3. Mathematical glyphs that came through as their plain-text fallback

- `≜` (definition equality) appeared as `,` in some places and as `≜` in others in the raw PDF text. In the screenshots it is consistently `≜`. Rendered as `\triangleq` throughout.
- `→` appeared cleanly in most places; in eq. 4 it shows as `−→` in the raw text but is unambiguously `→` in the screenshot.
- `∂` (partial derivative) came through correctly.
- `ϕ` vs `φ` — the paper uses `φ` (varphi) for the implicit function. In the raw text both `ϕ` and `φ` appear; the screenshot consistently uses the open-top varphi `φ`. Rendered as `\varphi`.

### 4. Subscripts on operators

The PDF text occasionally drops the math-mode subscript styling, producing strings like `Bθ` or `Bθ'` for $B^\theta$ and $B^{\theta'}$. In every case the screenshot showed it as a *superscript*, not a subscript — the soft Bellman operator is parameterized by $\theta$ as a superscript, $B^\theta$. This was corrected throughout.

Same issue in eq. 11: the raw text shows `Qw¯` for the target-network Q-value — this is $Q_{\bar w}$ (subscript $\bar w$). Confirmed against screenshot.

### 5. Eq. 14 — sign and structure

The raw PDF rendering of eq. 14 reads:

```
∂L^true(θ)/∂θ ≈ −  ∂L^true(w*)/∂w  ·  ∂²L(θ, w*)/(∂θ∂w)  |_{w*=φ(θ)}
```

The screenshot confirms the negative sign and the underbraces `grad Bellman` and `approx IFT`. This is preserved in the markdown.

A small note: the formula as written is a vector-times-Jacobian product. Under the IFT-with-identity-inverse approximation (paper Section 6, "approximate the inverse Jacobian term … with the identity matrix"), the actual derivation gives:

$$
\frac{\partial L^{\text{true}}}{\partial \theta}
= \frac{\partial L^{\text{true}}}{\partial w} \cdot \frac{\partial \varphi(\theta)}{\partial \theta}
\approx \frac{\partial L^{\text{true}}}{\partial w} \cdot \Big( - I \cdot \frac{\partial f}{\partial \theta} \Big)
= - \frac{\partial L^{\text{true}}}{\partial w} \cdot \frac{\partial^2 L(\theta, w^*)}{\partial \theta \, \partial w},
$$

since $f = \partial L / \partial w$ and $\partial f / \partial \theta = \partial^2 L / (\partial w \, \partial \theta)$. The minus sign in the printed eq. 14 is therefore correct and consistent with the IFT identity (eq. 7) when its leading minus is folded in.

### 6. Theorem 2 inequality bounds

The raw PDF line for the MLE bound has a dropped-superscript artifact:

```
≤ \frac{r}{1 - γ} + \frac{γprmax}{2(1 - γ)2}
```

The screenshot shows these as $\epsilon_r$ and $\epsilon_p$ (so the numerators are $\epsilon_r$ and $\gamma \epsilon_p r_{\max}$, not the bare letters $r$ and $p$), and the denominator is $(1-\gamma)^2$. Same issue with the OMD bound: `≤ /1−γ` is actually $\epsilon / (1 - \gamma)$. Both corrected in the markdown.

### 7. Figure 1, 3, 4, 5, 6, 7 plots

The screenshots of the plots are illegible at small sizes for fine-grained values (e.g. exact y-axis tick marks), but the *shape* and *legend* are clear. Where the markdown describes a figure, the description is taken from the figure caption, which is in the text and cross-checked against the screenshot.

For Figure 3 in particular, the legend in the screenshot uses the bound symbols:

$$
\frac{\epsilon}{1 - \gamma}, \quad \frac{\epsilon_r}{1 - \gamma} + \frac{\gamma \epsilon_p r_{\max}}{2(1 - \gamma)^2}
$$

These match Theorem 2.

### 8. Figure 2 MDP tuples

The MDP figure (page 4) labels each transition with a 4-tuple `(action, reward, transition probability, optimal Q value)`. The PDF text layer extracts these in scattered order; I reorganized them by source state in the markdown using the screenshot as ground truth. Verified values:

**Top (true MDP, Dadashi et al. 2019):**
- (0, −0.45, 0.7, **0.16**) — state 0, action 0, → state 0
- (0, −0.45, 0.3, **0.16**) — state 0, action 0, → state 1
- (1, −0.10, 0.99, **0.06**) — state 0, action 1, → state 0
- (1, −0.10, 0.01, **0.06**) — state 0, action 1, → state 1
- (0, 0.50, 0.8, **1.89**) — state 1, action 0, → state 1
- (0, 0.50, 0.2, **1.89**) — state 1, action 0, → state 0
- (1, 0.50, 0.01, **0.66**) — state 1, action 1, → state 1
- (1, 0.50, 0.99, **0.66**) — state 1, action 1, → state 0

**Bottom (OMD trained):** dynamics are deterministic.
- (0, 0.02, 1, **0.16**)
- (1, −1.64, 1, **0.06**)
- (0, 0.19, 1, **1.89**)
- (1, −1.04, 1, **0.66**)

Optimal Q values match between top and bottom MDPs by construction (both are $Q^*$-equivalent), so the bolded values are an internal consistency check that the OCR got the numbers right. They do match.

### 9. Eq. 10 (the limit)

PDF raw:
```
limα→0α logΣa' exp 1/α Q(s', a') = max_a' Q(s', a').
```

Confirmed against the screenshot: there are nested temperatures — `α log Σ exp (1/α) Q(s', a')`. The `1/α` inside the exp has to be there for the limit to recover `max`. Rendered in markdown as:

$$
\lim_{\alpha \to 0} \alpha \log \sum_{a'} \exp \frac{1}{\alpha} Q(s', a') = \max_{a'} Q(s', a').
$$

### 10. KL divergence definition (Section 4.2)

PDF raw:
```
DKL(p||pθ) = 1/(|S|·|A|) Σ_{s,a,s'} p(s'|s,a) log p(s'|s,a)/p_θ(s'|s,a)
```

Rendered with overline as in the screenshot ($\overline{D_{\text{KL}}}$, "average KL"). Confirmed.

### 11. Eq. 15 (VEP loss)

PDF raw extraction is fine; just flagging that $B^\theta_\pi V(s) = \mathbb{E}_{a \sim \pi(a|s),\, s' \sim p_\theta(s'|s,a)} \big( r_\theta(s,a) + \gamma V(s') \big)$ has the parenthesis grouping as written — the expectation is over both $a$ and $s'$, and the integrand is the bracketed sum. No change.

### 12. Algorithm 1 line ordering

The PDF text-layer extraction concatenates Algorithm 1's lines with mild reordering that breaks the loop structure (e.g. it places the inner `for` loop body inline before the `for` line in raw text). The screenshot confirms the canonical ordering, which is what's in the markdown. No content changes — just reflowed to the right structure.

---

## Items requiring no correction (clean in PDF text)

- Section structure (1 through 9 + Acknowledgements + References)
- All non-mathematical body text
- Figure captions (text content)
- Table 1 entries (MSE values)
- All in-line citation keys
- Reference list entries (with the diacritic exceptions noted in §2 above)

---

## Items NOT included in the working summary by user request

Per the user's instructions, the following were translated faithfully into the full `.md` mirror of the paper but were **excluded** from the `-summary.md`:

- Theorem 2 *proofs* (the paper itself only states the theorem and notes "the proof mostly follows derivations similar to the simulation lemma (Kearns and Singh 2002)" — there is no proof in the main body, so nothing was actually dropped).
- Proposition 1's proof sketch (also brief and mostly verbal in the original; main statement retained in summary).
- The full Related Work section (1.5 pages of contextual citations); summary keeps only what's needed to position OMD against MLE and VEP.
- The full Theoretical Analysis discussion around equivalence classes — summary keeps the key takeaway (the $Q^*$-equivalence class exists, OMD lands somewhere in it) but drops the state-abstraction tie-in.
- Most of the experimental discussion text — summary keeps only the practical hyperparameter table and the algorithm.

## Note about ISOM-2

The user's instructions reference "ISOM-2" as the algorithm to be implemented. This paper's algorithm is **OMD** (Algorithm 1: Model Based RL with OMD). ISOM-2 appears to be from one of the two earlier papers in the user's project queue, not this one. The summary therefore covers the full OMD algorithm at the level needed to implement it, with the understanding that OMD may be referenced as a baseline or upstream component when ISOM-2 is implemented from another paper.
