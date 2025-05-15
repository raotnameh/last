# # Assume:
# # student_probs: torch.Tensor of shape (B, T, C)
# # teacher_probs: torch.Tensor of shape (B, T, C)

# # 1. Get generated tokens by argmax over vocab
# # tokens: shape (B, T)
# tokens = student_probs.argmax(dim=-1)

# # 2. Gather student & teacher log-probs for those tokens
# # student_logp: (B, T)
# student_logp = torch.log(
#     student_probs.gather(-1, tokens.unsqueeze(-1))
# ).squeeze(-1)

# # teacher_logp: (B, T)
# teacher_logp = torch.log(
#     teacher_probs.gather(-1, tokens.unsqueeze(-1))
# ).squeeze(-1)

# # 3. Compute token-level rewards and baseline
# # reward = teacher log-prob (higher = better)
# # reward: (B, T)
# reward = teacher_logp

# # baseline: scalar baseline over batch & time
# # baseline: (1, 1)
# baseline = reward.mean(dim=(0, 1), keepdim=True)

# # 4. Token-level advantage: (B, T)
# advantage = reward - baseline

# # 5. RL loss per token (no reduction): (B, T)
# loss = -advantage * student_logp

# # 'loss' now holds the un-reduced per-token loss of shape (B, T)



        

import torch
import torch.nn.functional as F



def rl_loss_from_logits(student_logits, teacher_logits, tokens):
    """
    Compute per-token RL loss using log-softmax on logits and advantage weighting.

    Args:
    - student_logits (torch.Tensor): (B, T, C)
    - teacher_logits (torch.Tensor): (B, T, C)
    - TOkens : B,T

    Returns:
    - loss (torch.Tensor): (B, T), un-reduced per-token loss
    """

    # 2. Compute log-probabilities directly from logits
    student_log_probs = F.log_softmax(student_logits, dim=-1)  # (B, T, C)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)  # (B, T, C)

    # 3. Gather log-probs for sampled tokens: (B, T)
    student_logp = student_log_probs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    teacher_logp = teacher_log_probs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)

    # 4. Reward = teacher log-prob (B, T)
    reward = teacher_logp

    # 5. Baseline: scalar mean reward
    baseline = reward.mean(dim=(0, 1), keepdim=True)  # (1, 1)

    # 6. Advantage
    advantage = reward - baseline  # (B, T)

    # 7. RL loss
    loss = -advantage * student_logp  # (B, T)

    return loss
