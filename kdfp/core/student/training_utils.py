import torch
import torch.nn.functional as F

def jensen_shannon_divergence(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    beta: float,
    temperature: float,
    epsilon: float = 1e-8,
    mask_value: float = -1e8
) -> torch.Tensor:
    """
    Robust beta-weighted Jensen-Shannon Divergence loss for knowledge distillation
    with causal masking support.

    This implementation handles:
    - Masked positions in attention logits (e.g., -inf values).
    - Temperature scaling for softening distributions.
    - Proper T² compensation for correct gradient scaling in distillation.
    - Numerical stability for all calculations.

    Args:
        teacher_logits (torch.Tensor): Logits from the true distribution (P).
        student_logits (torch.Tensor): Logits from the approximating distribution (Q).
        beta (float): Weighting parameter between 0 and 1.
                      beta=0.5 is standard JSD.
                      beta > 0.5 weights KL(P||M) more.
                      beta < 0.5 weights KL(Q||M) more.
        temperature (float): Temperature for softening. Must be positive.
        epsilon (float): Small constant for numerical stability in log.
        mask_value (float): Value to replace -inf with for numerical stability.

    Returns:
        torch.Tensor: A single scalar loss value with proper temperature compensation.
    """
    # Step 1: Input Validation
    assert temperature > 0, "Temperature must be positive."
    assert 0 <= beta <= 1, "Beta must be between 0 and 1."
    assert teacher_logits.shape == student_logits.shape, "Logits must have the same shape."

    # Step 2: Detect masked positions BEFORE any modifications
    # A position is masked if it's -inf or a large negative number in either tensor.
    p_mask = torch.isinf(teacher_logits) | (teacher_logits <= mask_value * 0.9)
    q_mask = torch.isinf(student_logits) | (student_logits <= mask_value * 0.9)
    combined_mask = p_mask | q_mask

    # Step 3: Clean logits and apply temperature scaling
    p_clean = torch.where(torch.isinf(teacher_logits), mask_value, teacher_logits)
    q_clean = torch.where(torch.isinf(student_logits), mask_value, student_logits)
    
    scaled_p = p_clean / temperature
    scaled_q = q_clean / temperature

    # Step 4: Compute probabilities and log-probabilities
    # Using log_softmax is more numerically stable than softmax followed by log.
    p_log_probs = F.log_softmax(scaled_p, dim=-1)
    q_log_probs = F.log_softmax(scaled_q, dim=-1)
    
    p_probs = torch.exp(p_log_probs)
    q_probs = torch.exp(q_log_probs)

    # Step 5: Compute the mixture distribution M and its log-probabilities
    m_probs = beta * p_probs + (1 - beta) * q_probs
    m_log_probs = torch.log(m_probs.clamp(min=epsilon)) # Clamp to avoid log(0)

    # Step 6: Compute the two KL divergence terms element-wise
    # KL(P||M) = sum(P * (log(P) - log(M)))
    kl_p_m_elements = p_probs * (p_log_probs - m_log_probs)
    
    # KL(Q||M) = sum(Q * (log(Q) - log(M)))
    kl_q_m_elements = q_probs * (q_log_probs - m_log_probs)

    # Step 7: Combine into the final JSD formula element-wise
    jsd_elements = (beta * kl_p_m_elements) + ((1 - beta) * kl_q_m_elements)

    # Step 8: Zero out contributions from masked positions
    jsd_elements = jsd_elements.masked_fill(combined_mask, 0.0)

    # Step 9: Compute the mean loss over all unmasked elements
    num_unmasked = (~combined_mask).sum()

    if num_unmasked == 0:
        # Handle the edge case where every position is masked
        return torch.tensor(0.0, device=teacher_logits.device, dtype=jsd_elements.dtype)
        
    # The sum is over all dimensions (batch, sequence, vocab)
    loss = jsd_elements.sum() / num_unmasked

    # Step 10: Apply T² scaling for knowledge distillation
    # This ensures the gradients are correctly scaled with respect to temperature.
    loss = loss * (temperature ** 2)

    return loss
