import torch
import torch.nn as nn
from kdfp.utils.enums import ProjectionDirection

class RandomizedProjector(nn.Module):
    def __init__(self,
                 teacher_dim: int,
                 student_dim: int,
                 direction: ProjectionDirection,
                 torch_dtype: torch.dtype):
        super.__init__()
        self.teacher_dim = teacher_dim
        self.student_dum = student_dim
        self.direction = direction
        self.torch_dtype = torch_dtype


        if direction == ProjectionDirection.UP:
            random_matrix = torch.randn(student_dim, teacher_dim, dtype=torch_dtype)
        else:
            random_matrix = torch.randn(teacher_dim, student_dim, dtype=torch_dtype)

        # Perform QR decomposition to get the orthonormal matrix Q
        projection_matrix, _ = torch.linalg.qr(random_matrix)
        self.register_buffer('projection_matrix', projection_matrix)

    def forward(self, x):
        return torch.matmul(x, self.projection_matrix)