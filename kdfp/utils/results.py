from typing import Dict

def _format_params(num_params: int) -> str:
    """Formats a number into a human-readable string (e.g., 1.2M, 2.5B)."""
    if num_params >= 1_000_000_000:
        return f"{num_params / 1_000_000_000:.2f}B"
    elif num_params >= 1_000_000:
        return f"{num_params / 1_000_000:.2f}M"
    elif num_params >= 1_000:
        return f"{num_params / 1_000:.2f}K"
    else:
        return f"{num_params}"
    
def generate_result(name: str,
                    model,
                    task_name: str,
                    metric_name: str,
                    score: float) -> Dict:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_param_str = _format_params(total_params)
    training_param_str = _format_params(trainable_params)
    return {
        "EXPERIMENT": name,
        "TOTAL_PARAMS": total_param_str,
        "TRAINABLE_PARAMS": training_param_str,
        "TASK": task_name,
        "METRIC": metric_name,
        "SCORE": score,
    }