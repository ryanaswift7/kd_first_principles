
class ObjectiveWeights:
    def __init__(self, teacher_outputs, true_labels, embed, attn_scores, attn_out, mlp, resid):
        self.teacher_outputs = teacher_outputs
        self.true_labels = true_labels
        self.embed = embed
        self.attn_scores = attn_scores
        self.attn_out = attn_out
        self.mlp = mlp
        self.resid = resid

# reference this to 
OBJECTIVE_WEIGHT_VARIANTS_MAP = {
    "only_outputs": ObjectiveWeights(**{
        "teacher_outputs": 0.5,
        "true_labels": 0.5,
        "embed": 0,
        "attn_scores": 0,
        "attn_out": 0,
        "mlp": 0,
        "resid": 0,}),
    "balanced": ObjectiveWeights(**{
        "teacher_outputs": 0.143,
        "true_labels": 0.143,
        "embed": 0.143,
        "attn_scores": 0.143,
        "attn_out": 0.143,
        "mlp": 0.143,
        "resid": 0.143,}),
    "no_embed": ObjectiveWeights(**{
        "teacher_outputs": 0.166,
        "true_labels": 0.166,
        "embed": 0,
        "attn_scores": 0.166,
        "attn_out": 0.166,
        "mlp": 0.166,
        "resid": 0.166,}),
    "no_attn": ObjectiveWeights(**{
        "teacher_outputs": 0.2,
        "true_labels": 0.2,
        "embed": 0.2,
        "attn_scores": 0,
        "attn_out": 0,
        "mlp": 0.2,
        "resid": 0.2,}),
    "no_mlp": ObjectiveWeights(**{
        "teacher_outputs": 0.166,
        "true_labels": 0.166,
        "embed": 0.166,
        "attn_scores": 0.166,
        "attn_out": 0.166,
        "mlp": 0,
        "resid": 0.166,}),
    "no_resid": ObjectiveWeights(**{
        "teacher_outputs": 0.166,
        "true_labels": 0.166,
        "embed": 0.166,
        "attn_scores": 0.166,
        "attn_out": 0.166,
        "mlp": 0.166,
        "resid": 0,}),
    "with_embed": ObjectiveWeights(**{
        "teacher_outputs": 0.33,
        "true_labels": 0.33,
        "embed": 0.34,
        "attn_scores": 0,
        "attn_out": 0,
        "mlp": 0,
        "resid": 0,}),
    "with_attn": ObjectiveWeights(**{
        "teacher_outputs": 0.33,
        "true_labels": 0.33,
        "embed": 0,
        "attn_scores": 0.165,
        "attn_out": 0.165,
        "mlp": 0,
        "resid": 0,}),
    "with_mlp": ObjectiveWeights(**{
        "teacher_outputs": 0.33,
        "true_labels": 0.33,
        "embed": 0,
        "attn_scores": 0,
        "attn_out": 0,
        "mlp": 0.34,
        "resid": 0,}),
    "with_resid": ObjectiveWeights(**{
        "teacher_outputs": 0.33,
        "true_labels": 0.33,
        "embed": 0,
        "attn_scores": 0,
        "attn_out": 0,
        "mlp": 0,
        "resid": 0.34,}),
}
