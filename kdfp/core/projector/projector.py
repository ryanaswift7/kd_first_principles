import torch
import torch.nn as nn
import os
import json
from typing import Optional, Dict, Literal
from pytorch_lightning import LightningModule
from .create import create_with_raytune

"""
TODO: FIX THE MESS I JUST CREATED
    - forward passes
    - loss computation
    - train, val, and test steps
    - remove all default values
    - understand/fix new pl and ray components

"""

class Projector(LightningModule):
    def __init__(self, teacher_dim: int, student_dim: int, lr: float, weight_decay: float,
                 loss_weights: Dict[Literal["reconstruction", "alignment"], float], tied_weights: bool = False):
        super().__init__()
        self.create = classmethod(create_with_raytune)
        self.save_hyperparameters()
        
        self.teacher_dim = teacher_dim
        self.student_dim = student_dim
        self.tied_weights = tied_weights
        
        # Encoder: teacher -> student space (i.e. DOWN)
        self.encoder = nn.Linear(teacher_dim, student_dim)
        
        # Decoder: student -> teacher space (i.e. UP)
        self.decoder = nn.Linear(student_dim, teacher_dim)
        
        # Weight tying if enabled
        if tied_weights:
            self.decoder.weight = nn.Parameter(self.encoder.weight.t())
        
        # Default loss weights
        self.loss_weights = loss_weights
        
        self.criterion = nn.MSELoss()

    def forward(self, x, direction: Optional[str] = None):
        """
        Forward pass with direction control
        
        Args:
            x: input tensor
            direction: 
                - "autoencode": teacher -> student -> teacher reconstruction
                - "encode": teacher -> student projection only
                - "decode": student -> teacher projection only
                - "align": student -> teacher alignment (for inference)
        """
        if direction == None:
            # Full autoencoder: teacher -> student -> teacher
            latent = self.encoder(x)
            reconstructed = self.decoder(latent)
            return reconstructed, latent
        
        elif direction == "down":
            # Teacher -> student projection
            return self.encoder(x)
        
        elif direction == "up":
            # Student -> teacher projection  
            return self.decoder(x)
        
        else:
            raise ValueError(f"Unknown direction: {direction}")

    def compute_losses(self, X_teacher, X_student):
        """Compute all loss components for multi-objective training"""
        
        # Autoencoder reconstruction loss
        reconstructed_teacher, latent_student = self(X_teacher, direction=None)
        recon_loss = self.criterion(reconstructed_teacher, X_teacher)
        
        # Representation alignment loss (student -> teacher space)
        align_loss = self.criterion(latent_student, X_student)
        
        # Total weighted loss
        total_loss = (
            self.loss_weights['reconstruction'] * recon_loss +
            self.loss_weights['alignment'] * align_loss
        )
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'align_loss': align_loss,
        }

    def training_step(self, batch, batch_idx):
        X_teacher, X_student = batch
        losses = self.compute_losses(X_teacher, X_student)
        
        # Log all losses
        self.log('train_total_loss', losses['total_loss'], prog_bar=True)
        self.log('train_recon_loss', losses['recon_loss'])
        self.log('train_align_loss', losses['align_loss'])
        
        return losses['total_loss']

    def validation_step(self, batch, batch_idx):
        X_teacher, X_student = batch
        losses = self.compute_losses(X_teacher, X_student)
        
        self.log('val_total_loss', losses['total_loss'], prog_bar=True)
        self.log('val_recon_loss', losses['recon_loss'])
        self.log('val_align_loss', losses['align_loss'])
        
        return losses['total_loss']

    def test_step(self, batch, batch_idx):
        X_teacher, X_student = batch
        
        # Compute all losses
        losses = self.compute_losses(X_teacher, X_student)
        
        # Additional evaluation metrics
        with torch.no_grad():
            # Reconstruction quality
            reconstructed, latent = self(X_teacher, direction=None)
            recon_mae = nn.L1Loss()(reconstructed, X_teacher)
            recon_cosine = torch.nn.functional.cosine_similarity(reconstructed, X_teacher, dim=-1).mean()
            
            # Alignment quality
            align_mae = nn.L1Loss()(latent, X_student)
            align_cosine = torch.nn.functional.cosine_similarity(latent, X_student, dim=-1).mean()
            
        # Log all metrics
        metrics = {
            **losses,
            'recon_mae': recon_mae,
            'recon_cosine': recon_cosine,
            'align_mae': align_mae,
            'align_cosine': align_cosine,
        }
        
        for name, value in metrics.items():
            self.log(f'test_{name}', value)
        
        return metrics

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay
        )

    # def project(self, x, projection_direction: str):
    #     """Public method for projection in specific direction"""
    #     if projection_direction == "teacher_to_student":
    #         return self(x, direction="encode")
    #     elif projection_direction == "student_to_teacher":
    #         return self(x, direction="decode")
    #     elif projection_direction == "autoencode":
    #         return self(x, direction="autoencode")
    #     else:
    #         raise ValueError(f"Invalid projection direction: {projection_direction}")

    def save_pretrained(self, save_path: str):
        """Save model with configuration"""
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        torch.save(self.state_dict(), save_path)
        
        config_path = save_path + '_config.json'
        config = {
            'teacher_dim': self.teacher_dim,
            'student_dim': self.student_dim,
            'tied_weights': self.tied_weights,
            'lr': self.hparams.lr,
            'weight_decay': self.hparams.weight_decay,
            'loss_weights': self.loss_weights
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)


# Add as class method
# Projector.create = classmethod(create_with_raytune)