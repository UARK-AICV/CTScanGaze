"""
CT-Searcher: 3D Scanpath Prediction for CT Volumes
Simplified implementation based on the CT-ScanGaze paper (ICCV 2025)
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from models.positional_encodings import PositionEmbeddingSine2d


class CTSearcher(nn.Module):
    """
    CT-Searcher model for 3D volumetric scanpath prediction.
    
    Architecture (as described in paper sections 4.1-4.5):
    1. Feature Extraction (4.1): Pre-extracted features from frozen Swin UNETR
    2. Transformer Decoder (4.2): Processes features with learnable queries
    3. Spatial Prediction (4.3): Predicts 3D fixation locations
    4. Duration Prediction (4.4): Predicts fixation durations
    """
    
    def __init__(
        self,
        d_model=768,
        nhead=8,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        spatial_dim=(8, 8, 8),
        max_length=400,
        device="cuda:0",
    ):
        super(CTSearcher, self).__init__()
        self.d_model = d_model
        self.spatial_dim = spatial_dim
        self.max_length = max_length
        self.device = device
        
        # Input projection for Swin UNETR features
        self.input_proj = nn.Linear(768, d_model)
        
        # Learnable query embeddings (positional embeddings for fixation sequence)
        self.query_embed = nn.Embedding(max_length, d_model)
        
        # Positional encoding for spatial features
        self.pos_embed = PositionEmbeddingSine2d(
            spatial_dim[:2], hidden_dim=d_model, normalize=True, device=device
        )
        
        # Transformer Decoder (Section 4.2)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=False,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
        )
        
        # Spatial Prediction Head (Section 4.3)
        # Predicts discrete spatial locations in 3D (x, y, z) + termination token
        self.spatial_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, spatial_dim[0] * spatial_dim[1] * spatial_dim[2] + 1)
        )
        
        # Duration Prediction Head (Section 4.4)
        # Predicts log-normal distribution parameters (mu, sigma) for fixation duration
        self.duration_mu = nn.Linear(d_model, 1)
        self.duration_logvar = nn.Linear(d_model, 1)
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        """Initialize parameters"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        src: Tensor,
        tgt_mask: Optional[Tensor] = None,
    ):
        """
        Args:
            src: Pre-extracted features from Swin UNETR [batch_size, seq_len, feature_dim]
            tgt_mask: Causal mask for autoregressive decoding
            
        Returns:
            Dictionary containing:
            - 'spatial_logits': Spatial prediction logits [batch_size, max_length, num_locations]
            - 'duration_mu': Mean of log-normal distribution [batch_size, max_length]
            - 'duration_sigma2': Variance of log-normal distribution [batch_size, max_length]
        """
        batch_size = src.size(0)
        
        # Project input features
        src = self.input_proj(src)  # [batch, seq_len, d_model]
        
        # Add positional encoding to memory
        src = src.permute(1, 0, 2)  # [seq_len, batch, d_model]
        
        # Prepare decoder queries
        tgt = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)  # [max_length, batch, d_model]
        
        # Generate causal mask for autoregressive decoding
        if tgt_mask is None:
            tgt_mask = self._generate_square_subsequent_mask(self.max_length).to(src.device)
        
        # Transformer Decoder (Section 4.2)
        decoder_output = self.transformer_decoder(
            tgt=tgt,
            memory=src,
            tgt_mask=tgt_mask,
        )  # [max_length, batch, d_model]
        
        decoder_output = decoder_output.permute(1, 0, 2)  # [batch, max_length, d_model]
        
        # Spatial Prediction (Section 4.3)
        spatial_logits = self.spatial_head(decoder_output)  # [batch, max_length, num_locations+1]
        
        # Duration Prediction (Section 4.4)
        duration_mu = self.duration_mu(decoder_output).squeeze(-1)  # [batch, max_length]
        duration_sigma2 = torch.exp(self.duration_logvar(decoder_output)).squeeze(-1)  # [batch, max_length]
        
        return {
            'spatial_logits': spatial_logits,
            'duration_mu': duration_mu,
            'duration_sigma2': duration_sigma2,
        }
    
    def _generate_square_subsequent_mask(self, sz: int) -> Tensor:
        """Generate causal mask for autoregressive decoding"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def inference(self, src: Tensor):
        """
        Inference mode with softmax activation for spatial predictions
        
        Args:
            src: Pre-extracted features from Swin UNETR
            
        Returns:
            Dictionary with predictions including softmax probabilities
        """
        outputs = self.forward(src)
        outputs['spatial_probs'] = F.softmax(outputs['spatial_logits'], dim=-1)
        return outputs

