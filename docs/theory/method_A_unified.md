# Method A: Unified Mixture GRPO

## Core Idea

Combine online student rollouts and offline teacher rollouts into a **single unified group** for advantage computation and loss calculation. No separate weighting — all completions are treated equally within the group.

---

## Algorithm

For each prompt $x_b$ in a training batch:

### 1. Online Sampling

Sample $G$ completions from the current student policy:

$$
a_{b,1}^{stu}, \dots, a_{b,G}^{stu} \sim \pi_{old}(\cdot | x_b)
$$

Record student behavior log-probabilities: $\log \pi_{old}(a_{b,g}^{stu} | x_b)$.

### 2. Retrieve Teacher Rollouts

Look up $K$ pre-computed teacher completions for the same prompt:

$$
a_{b,1}^{tea}, \dots, a_{b,K}^{tea}
$$

with stored teacher log-probabilities: $\log \pi_T(a_{b,k}^{tea} | x_b)$.

### 3. Reward Computation

Assign binary reward to all $G + K$ completions:

$$
r_i = \begin{cases} 1 & \text{if final answer is correct} \\ 0 & \text{otherwise} \end{cases}
$$

### 4. Unified Group Advantage

Pool all $G + K$ completions into one group. Compute group statistics:

$$
\mu_b = \frac{1}{G + K} \sum_{i=1}^{G+K} r_i
$$

$$
s_b = \sqrt{\frac{1}{G+K} \sum_{i=1}^{G+K} (r_i - \mu_b)^2}
$$

Advantage for each completion:

$$
A_i = \frac{r_i - \mu_b}{\max(s_b, \epsilon)}
$$

### 5. Importance Ratios

For student completions (on-policy, ratio $\approx 1$):

$$
\rho_{b,g}^{stu} = \frac{\pi_\theta(a_{b,g}^{stu} | x_b)}{\pi_{old}(a_{b,g}^{stu} | x_b)}
$$

For teacher completions (off-policy):

$$
\rho_{b,k}^{tea} = \frac{\pi_\theta(a_{b,k}^{tea} | x_b)}{\pi_T(a_{b,k}^{tea} | x_b)}
$$

### 6. Clipped Surrogate Loss

All $G + K$ completions use the same PPO clipped objective:

$$
\ell_i = \min(\rho_i A_i, \; \text{clip}(\rho_i, 1-\varepsilon, 1+\varepsilon) A_i)
$$

$$
L_{sur} = -\frac{1}{m} \sum_{b=1}^{m} \frac{1}{G+K} \sum_{i=1}^{G+K} \ell_i
$$

### 7. KL Regularization

$$
L_{total} = L_{sur} + \beta \cdot L_{KL}
$$

Where $L_{KL}$ is computed against a reference model (base model or lagged snapshot).

---

## Key Properties

- **No separate weighting hyperparameter**: teacher and student completions contribute equally within the group. The relative influence is determined implicitly by the number of completions ($G$ vs $K$).
- **Shared baseline**: the advantage baseline $\mu_b$ naturally reflects both student and teacher performance levels.
- **Mixed behavior policies**: the batch contains data from two different behavior policies ($\pi_{old}$ and $\pi_T$), each with its own importance ratio denominator.
- **Simpler implementation**: only one loss computation, no $\lambda$ to tune.

---

## Comparison with Offline GRPO

| Aspect | Offline GRPO | Unified Mixture |
|--------|-------------|-----------------|
| Student rollouts | None | Online, every step |
| Teacher rollouts | Only data source | Supplementary |
| Advantage baseline | Teacher group mean | Mixed group mean |
| Exploration | None | Student explores |
| Behavior policy | Teacher only | Mixed (student + teacher) |

---

## Hyperparameters

| Parameter | Typical value |
|-----------|---------------|
| Student group size $G$ | 4-8 |
| Teacher completions $K$ | 1-4 |
| PPO clip $\varepsilon$ | 0.2 |
| KL coefficient $\beta$ | small (tuned) |
| Reference sync steps | 50-200 |
