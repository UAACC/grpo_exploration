# Method B: Weighted Mixture GRPO

## Core Idea

Combine online student rollouts and offline teacher rollouts with **separate loss terms**, weighted by a hyperparameter $\lambda$. The teacher's advantage is normalized using the **student's online group statistics**, linking the teacher signal to the student's current ability.

---

## Algorithm

For each prompt $x_b$ in a training batch:

### 1. Online Sampling

Sample $G$ completions from the current student policy:

$$
a_{b,1}^{stu}, \dots, a_{b,G}^{stu} \sim \pi_{old}(\cdot | x_b)
$$

Record student behavior log-probabilities: $\log \pi_{old}(a_{b,g}^{stu} | x_b)$.

### 2. Reward Computation (Online)

Assign binary reward to each student completion:

$$
r_{b,g} \in \{0, 1\}
$$

### 3. Online Group Statistics

Compute mean and scale from **student completions only**:

$$
\mu_b = \frac{1}{G} \sum_{g=1}^{G} r_{b,g}
$$

$$
s_b = \sqrt{\frac{1}{G} \sum_{g=1}^{G} (r_{b,g} - \mu_b)^2}
$$

### 4. Online Advantages

$$
A_{b,g}^{on} = \frac{r_{b,g} - \mu_b}{\max(s_b, \epsilon)}
$$

### 5. Offline Teacher Advantage

Retrieve teacher rollout for the same prompt. Teacher reward $r_b^T$.

Normalize using the **online (student) group statistics**:

$$
A_b^{off} = \frac{r_b^T - \mu_b}{\max(s_b, \epsilon)}
$$

This is the key design: the teacher signal is measured relative to what the student can currently achieve.

### 6. Importance Ratios

Online (on-policy, ratio $\approx 1$):

$$
\rho_{b,g}^{on} = \frac{\pi_\theta(a_{b,g}^{stu} | x_b)}{\pi_{old}(a_{b,g}^{stu} | x_b)}
$$

Offline (off-policy):

$$
\rho_b^{off} = \frac{\pi_\theta(a_b^T | x_b)}{\pi_T(a_b^T | x_b)}
$$

### 7. Clipped Surrogate Losses

Online loss:

$$
\ell_{b,g}^{on} = \min(\rho_{b,g}^{on} A_{b,g}^{on}, \; \text{clip}(\rho_{b,g}^{on}, 1-\varepsilon, 1+\varepsilon) A_{b,g}^{on})
$$

Offline loss:

$$
\ell_b^{off} = \min(\rho_b^{off} A_b^{off}, \; \text{clip}(\rho_b^{off}, 1-\varepsilon, 1+\varepsilon) A_b^{off})
$$

### 8. Total Objective

$$
L_{sur} = -\frac{1}{m} \sum_{b=1}^{m} \frac{1}{G} \sum_{g=1}^{G} \ell_{b,g}^{on} \;-\; \lambda \frac{1}{m} \sum_{b=1}^{m} \ell_b^{off}
$$

$$
L_{total} = L_{sur} + \beta \cdot L_{KL}
$$

---

## Key Properties

- **Explicit teacher weighting**: $\lambda$ directly controls how much the teacher influences training. Can be tuned or annealed.
- **Student-relative baseline**: teacher advantage uses the student's online $\mu_b$, so:
  - If student already solves the problem ($\mu_b$ high), a correct teacher completion has low advantage.
  - If student struggles ($\mu_b$ low), a correct teacher completion has high advantage.
- **Separate loss terms**: online and offline losses are computed independently, avoiding interaction effects in advantage normalization.
- **Teacher doesn't affect student baseline**: the online group statistics ($\mu_b$, $s_b$) come purely from student rollouts, preventing the teacher from distorting the student's self-assessment.

---

## Comparison with Method A (Unified)

| Aspect | Method A (Unified) | Method B (Weighted) |
|--------|-------------------|-------------------|
| Advantage baseline | Mixed (student + teacher) | Student only |
| Teacher influence | Implicit (via $K/(G+K)$) | Explicit ($\lambda$) |
| Loss computation | Single unified loss | Two separate losses |
| Teacher in baseline | Yes | No |
| Tuning flexibility | Less (adjust $K$) | More (adjust $\lambda$) |

---

## Hyperparameters

| Parameter | Typical value |
|-----------|---------------|
| Online group size $G$ | 4-8 |
| PPO clip $\varepsilon$ | 0.2 |
| Offline weight $\lambda$ | 0.1-0.5 |
| KL coefficient $\beta$ | small (tuned) |
| Reference sync steps | 50-200 |

---

## Failure Modes

### Teacher Dominance
Large $\lambda$ causes the model to imitate the teacher rather than explore.
Mitigation: reduce $\lambda$, increase $G$.

### No Online Reward Variance
If all student rollouts are incorrect, $s_b = 0$ and advantages collapse.
Mitigation: increase temperature, increase $G$, start from stronger checkpoint.

### Ratio Explosion
Off-policy teacher ratio can become very large.
Mitigation: PPO clipping, KL regularization.
