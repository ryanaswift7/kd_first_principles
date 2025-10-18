import torch
import torch.nn as nn
from kdfp.utils.enums import ProjectionDirection
import torch.nn.functional as F

class RandomizedProjector(nn.Module):
    def __init__(self,
                 teacher_dim: int,
                 student_dim: int,
                 direction: ProjectionDirection,
                 torch_dtype: torch.dtype):
        super(RandomizedProjector, self).__init__()
        self.teacher_dim = teacher_dim
        self.student_dum = student_dim
        self.direction = direction
        self.torch_dtype = torch_dtype

        """
        It only makes sense to construct student_dim orthonormal vectors
        from the teacher_dim space, and not the other way around
        (because is it impossible for there to be more mutually orthogonal
        vectors than there are dimensions).

        Thus only the JL Lemma only holds for the down projection. I am also
        trying the upward projection, though the same theoretical guarantees
        do not apply. For that, I simply transpose the Q matrix from QR decomp
        """
        random_matrix = torch.randn(teacher_dim, student_dim)
        # Perform QR decomposition to get the orthonormal matrix Q
        projection_matrix, _ = torch.linalg.qr(random_matrix)
        projection_matrix = projection_matrix.to(torch_dtype)

        if direction == ProjectionDirection.UP:
            projection_matrix = projection_matrix.T

        self.register_buffer('projection_matrix', projection_matrix)

    def forward(self, x):
        # F.linear has an implicit transpose, so I have to transpose it back
        return F.linear(x, self.projection_matrix.T)