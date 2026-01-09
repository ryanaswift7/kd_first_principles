import math

from pydantic import Field, model_validator

from kdfp.schemas.configBase import ConfigBase


class ObjectiveWeights(ConfigBase):
    teacher_outputs: float = Field(
        ..., description="Weight for teacher outputs objective", ge=0
    )
    true_labels: float = Field(
        ..., description="Weight for true labels objective", ge=0
    )
    embed: float = Field(..., description="Weight for embedding layer objective", ge=0)
    attn_scores: float = Field(
        ..., description="Weight for attention scores objective", ge=0
    )
    attn_out: float = Field(
        ..., description="Weight for attention output objective", ge=0
    )
    mlp: float = Field(..., description="Weight for MLP layer objective", ge=0)
    resid: float = Field(
        ..., description="Weight for residual connection objective", ge=0
    )

    @model_validator(mode="after")
    def check_weights_sum(self) -> "ObjectiveWeights":
        total = sum(
            [
                self.teacher_outputs,
                self.true_labels,
                self.embed,
                self.attn_scores,
                self.attn_out,
                self.mlp,
                self.resid,
            ]
        )

        # Check if sum is approximately 1.0, accounting for floating point errors
        if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(
                f"Weights must sum to 1.0, but got {total:.12f}. "
                f"Difference: {abs(total - 1.0):.2e}. "
                "Consider normalizing your weights."
            )

        return self


# reference this to
OBJECTIVE_WEIGHT_VARIANTS_MAP = {
    "only_outputs": ObjectiveWeights(
        **{
            "teacher_outputs": 1 / 2,
            "true_labels": 1 / 2,
            "embed": 0 / 2,
            "attn_scores": 0 / 2,
            "attn_out": 0 / 2,
            "mlp": 0 / 2,
            "resid": 0 / 2,
        }
    ),
    "balanced": ObjectiveWeights(
        **{
            "teacher_outputs": 1 / 7,
            "true_labels": 1 / 7,
            "embed": 1 / 7,
            "attn_scores": 1 / 7,
            "attn_out": 1 / 7,
            "mlp": 1 / 7,
            "resid": 1 / 7,
        }
    ),
    "no_embed": ObjectiveWeights(
        **{
            "teacher_outputs": 1 / 6,
            "true_labels": 1 / 6,
            "embed": 0 / 6,
            "attn_scores": 1 / 6,
            "attn_out": 1 / 6,
            "mlp": 1 / 6,
            "resid": 1 / 6,
        }
    ),
    "no_attn_scores": ObjectiveWeights(
        **{
            "teacher_outputs": 1 / 6,
            "true_labels": 1 / 6,
            "embed": 1 / 6,
            "attn_scores": 0 / 6,
            "attn_out": 1 / 6,
            "mlp": 1 / 6,
            "resid": 1 / 6,
        }
    ),
    "no_attn_out": ObjectiveWeights(
        **{
            "teacher_outputs": 1 / 6,
            "true_labels": 1 / 6,
            "embed": 1 / 6,
            "attn_scores": 1 / 6,
            "attn_out": 0 / 6,
            "mlp": 1 / 6,
            "resid": 1 / 6,
        }
    ),
    "no_mlp": ObjectiveWeights(
        **{
            "teacher_outputs": 1 / 6,
            "true_labels": 1 / 6,
            "embed": 1 / 6,
            "attn_scores": 1 / 6,
            "attn_out": 1 / 6,
            "mlp": 0 / 6,
            "resid": 1 / 6,
        }
    ),
    "no_resid": ObjectiveWeights(
        **{
            "teacher_outputs": 1 / 6,
            "true_labels": 1 / 6,
            "embed": 1 / 6,
            "attn_scores": 1 / 6,
            "attn_out": 1 / 6,
            "mlp": 1 / 6,
            "resid": 0 / 6,
        }
    ),
    "with_embed": ObjectiveWeights(
        **{
            "teacher_outputs": 1 / 3,
            "true_labels": 1 / 3,
            "embed": 1 / 3,
            "attn_scores": 0 / 3,
            "attn_out": 0 / 3,
            "mlp": 0 / 3,
            "resid": 0 / 3,
        }
    ),
    "with_attn_scores": ObjectiveWeights(
        **{
            "teacher_outputs": 1 / 3,
            "true_labels": 1 / 3,
            "embed": 0 / 3,
            "attn_scores": 1 / 3,
            "attn_out": 0 / 3,
            "mlp": 0 / 3,
            "resid": 0 / 3,
        }
    ),
    "with_attn_out": ObjectiveWeights(
        **{
            "teacher_outputs": 1 / 3,
            "true_labels": 1 / 3,
            "embed": 0 / 3,
            "attn_scores": 0 / 3,
            "attn_out": 1 / 3,
            "mlp": 0 / 3,
            "resid": 0 / 3,
        }
    ),
    "with_mlp": ObjectiveWeights(
        **{
            "teacher_outputs": 1 / 3,
            "true_labels": 1 / 3,
            "embed": 0 / 3,
            "attn_scores": 0 / 3,
            "attn_out": 0 / 3,
            "mlp": 1 / 3,
            "resid": 0 / 3,
        }
    ),
    "with_resid": ObjectiveWeights(
        **{
            "teacher_outputs": 1 / 3,
            "true_labels": 1 / 3,
            "embed": 0 / 3,
            "attn_scores": 0 / 3,
            "attn_out": 0 / 3,
            "mlp": 0 / 3,
            "resid": 1 / 3,
        }
    ),
    "finetuning": ObjectiveWeights(
        **{
            "teacher_outputs": 0.0,
            "true_labels": 1.0,
            "embed": 0.0,
            "attn_scores": 0.0,
            "attn_out": 0.0,
            "mlp": 0.0,
            "resid": 0.0,
        }
    ),
}
