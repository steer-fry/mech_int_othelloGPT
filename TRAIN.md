**Model architecture**
- Freeze the 8-layer OthelloGPT, including embeddings and output head.
- After each block, edit only the final input token’s residual vector:
  $\tilde{x}^{\ell}_t = x^{\ell}_t + g_\ell(x^{\ell}_t,r)$.
- Use a separate editor per layer:
  $g_\ell(x,r)=U_\ell\,\mathrm{ReLU}(D_\ell[x;r]+b_\ell)$.
- Encode the rule as $r=[\mathbf{1}_F;\mathbf{1}_E]\in\{0,1\}^{128}$:
  two 64-square masks specifying FLIP and ERASE. Use the simulator’s square ordering: A1, A2, ..., A8, B1, ..., H8.
- With bottleneck width 128:
  $D_\ell\in\mathbb{R}^{128\times640}$,
  $U_\ell\in\mathbb{R}^{512\times128}$,
  $b_\ell\in\mathbb{R}^{128}$.

**Training initialization**
- Initialize $D_\ell$ randomly; initialize $U_\ell=0$ and $b_\ell=0$.
- The edited model initially matches the original model exactly.
- Train all editors jointly through the actual edited forward pass.
- Keep gradients enabled through frozen transformer blocks.
- Bypass editors when no rule is supplied.

**Loss**
- Let $G$ be the legal next moves on the simulator’s edited board.
- Use uniform legal-move cross-entropy:
  $\mathcal{L}=-\frac{1}{|G|}\sum_{a\in G}\log p_\Phi(a\mid m_{<t},r)$.
- Compute softmax over the full vocabulary, including illegal tokens.
- Average over the batch and update only editor parameters.

**Dataset**
- Reuse the probe-reproduction game sequences and game-level splits.
- Sample a game prefix; reconstruct its board and side to move.
- Choose FLIP or ERASE with equal probability before sampling a candidate.
- Choose a changed-legal-set target with probability 90%, or an unchanged
  control with probability 10%, independently of the action.
- Propose a training game uniformly, a prefix length uniformly from 1–59,
  and one occupied noncentral square uniformly. Retry candidates within the
  chosen action and target stratum until the target is nonterminal and its
  legal-set change status matches. Compare simulator legal moves before and
  after the edit; preserve the same turn and implicit-pass convention.
- Apply the edit once, preserve the turn, and compute $G$.
- Each example is: original prefix, rule encoding, edited legal-move target.
- Follow the simulator’s pass convention; skip terminal targets.
- Use fixed validation/test examples from their respective game splits.
  Retain the original seeded examples for comparison with the width-32 run.
  Report changed and unchanged target strata separately.

**Starting hyperparameters — to tune**
- Bottleneck: 128; activation: ReLU; batch size: 64.
- Optimizer: AdamW; learning rate: 3e-4; weight decay: 1e-4.
- Constant learning rate; seed: 42.
- Validate every 500 updates; maximum 10,000 updates.
- Early stopping: 10 validations without improvement.
- Save the lowest-validation-loss checkpoint.
- Report top-$|G|$ legal-set accuracy and false positives/negatives.
