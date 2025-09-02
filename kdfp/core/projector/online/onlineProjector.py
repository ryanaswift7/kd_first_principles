import torch.nn as nn
from typing import Literal

class OnlineProjector(nn.Module):
    def __init__(self, teacher_dim: int, student_dim: int, direction: Literal['up', 'down']):
        super.__init__()
        self.teacher_dim = teacher_dim
        self.student_dum = student_dim
        self.direction = direction

        if direction == 'up':
            self.project = nn.Linear(student_dim, teacher_dim, bias=False)
        else:
            self.project = nn.Linear(teacher_dim, student_dim, bias=False)

    def forward(self, x):
        return self.project(x)