import torch
import torch.nn.functional as F

def rl_loss_from_logits(student_logits, teacher_logits, tokens, mask, sent):
    """
    Compute per-token REINFORCE loss for a student LM using
    teacher log-probs as rewards, with masking.

    Args:
        student_logits (Tensor): [B, T, C]
        teacher_logits (Tensor): [B, T, C]
        tokens         (LongTensor): [B, T]  sampled token indices
        mask           (Tensor): [B, T] float mask (1=keep, 0=ignore)
        sent           (List[str]): [B] original input sentences

    Returns:
        per_token_loss (Tensor): [B, T] un-reduced loss (zeroed where mask=0)
    """
    # 1) Log‐probs over full vocab
    student_log_probs = F.log_softmax(student_logits, dim=-1)  # [B, T, C]
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)  # [B, T, C]

    # 2) Gather log‐probs at sampled tokens
    student_logp = student_log_probs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)  # [B, T]
    teacher_logp = teacher_log_probs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)  # [B, T]

    # 3) Reward = teacher log‐prob
    reward = teacher_logp  # [B, T]

    # 4) Compute baseline over only the unmasked tokens
    total_mask = mask.sum()
    # avoid division by zero
    if total_mask > 0:
        baseline = (reward * mask).sum() / total_mask  # scalar
    else:
        baseline = 0.0

    # 5) Advantage and masking
    advantage = (reward - baseline) * mask  # [B, T]

    # 6) Per‐token REINFORCE loss (zero where mask=0)
    per_token_loss = -advantage * student_logp  # [B, T]
    per_token_loss = per_token_loss * mask      # just in case
    
    # 7) Print per-sentence advantages
    for i in range(len(sent)):
        # get only the valid positions
        valid_positions = mask[i].bool()
        adv_vals = advantage[i][valid_positions].tolist()
        print(f"Sentence {i}: {sent[i]}")
        print(f"  Advantages: {adv_vals}")


    return per_token_loss
