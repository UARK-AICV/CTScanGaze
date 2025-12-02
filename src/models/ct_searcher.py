"""
CT-Searcher: 3D Scanpath Prediction for CT Volumes
Implementation based on the CT-ScanGaze paper (ICCV 2025)
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from models.positional_encodings import PositionEmbeddingSine3d


class CTSearcher(nn.Module):
    """
    CT-Searcher model for 3D volumetric scanpath prediction.
    
    Architecture (as described in paper sections 4.1-4.5):
    1. Feature Extraction (4.1): 
       - Pre-extracted features from frozen Swin UNETR (visual encoder)
       - MLP projection with learnable stop token τ
       - 3D Positional Encoding
       - Transformer Encoder to create 3D-aware representations E(Z)
    2. Transformer Decoder (4.2): 
       - 3D PE applied again on E(Z)
       - Learnable queries Q attend to E(Z) to produce R
    3. Spatial Prediction (4.3): 
       - Ŷ = softmax(FC(R) ⊗ E(Z)^T)
    4. Duration Prediction (4.4): 
       - Predicts log-normal distribution parameters (μ, σ)
    """
    
    def __init__(
        self,
        d_model=768,
        nhead=8,
        num_encoder_layers=6,
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
        
        # Spatial tokens count (H' * W' * D')
        self.num_spatial_tokens = spatial_dim[0] * spatial_dim[1] * spatial_dim[2]
        
        # Input projection MLP for Swin UNETR features
        self.input_proj = nn.Sequential(
            nn.Linear(768, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        
        # Learnable stop token τ (Section 4.1)
        # This token represents 'stop' fixation and is concatenated with features
        self.stop_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Learnable query embeddings Q for decoder (Section 4.2)
        self.query_embed = nn.Embedding(max_length, d_model)
        
        # 3D Positional Encoding (Section 4.1)
        self.pos_embed = PositionEmbeddingSine3d(
            spatial_dim[:2], hidden_dim=d_model, normalize=True, device=device
        )
        
        # Transformer Encoder (Section 4.1)
        # Composes 3D-aware representations E(Z) from Z + PE
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )
        
        # Transformer Decoder (Section 4.2)
        # Uses learnable queries Q to attend to E(Z)
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
        # FC layer to project decoder output R before multiplying with E(Z)
        # Ŷ = softmax(FC(R) ⊗ E(Z)^T)
        self.spatial_fc = nn.Linear(d_model, d_model)
        
        # Duration Prediction Head (Section 4.4)
        # Predicts log-normal distribution parameters (mu, sigma) for fixation duration
        self.duration_mu = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )
        self.duration_logvar = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        """Initialize parameters"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # Initialize stop token
        nn.init.normal_(self.stop_token, std=0.02)
    
    def forward(
        self,
        src: Tensor,
        tgt_mask: Optional[Tensor] = None,
    ):
        """
        Args:
            src: Pre-extracted features from Swin UNETR [batch_size, seq_len, feature_dim]
                 seq_len should be H' * W' * D' (e.g., 8*8*8 = 512)
            tgt_mask: Causal mask for parallel decoding
            
        Returns:
            Dictionary containing:
            - 'spatial_logits': Spatial prediction logits [batch_size, max_length, num_locations+1]
            - 'duration_mu': Mean of log-normal distribution [batch_size, max_length]
            - 'duration_sigma2': Variance of log-normal distribution [batch_size, max_length]
        """
        batch_size = src.size(0)
        
        # === Feature Extraction (Section 4.1) ===
        
        # Project input features through MLP
        z = self.input_proj(src)  # [batch, seq_len, d_model]
        
        # Concatenate learnable stop token τ: Z = MLP(cc(F, τ))
        stop_tokens = self.stop_token.expand(batch_size, 1, -1)  # [batch, 1, d_model]
        z = torch.cat([z, stop_tokens], dim=1)  # [batch, seq_len+1, d_model]
        
        # Permute for transformer: [seq_len+1, batch, d_model]
        z = z.permute(1, 0, 2)
        
        # Apply 3D positional encoding on spatial tokens only (not on stop token)
        # PE is applied on tokens 0 to num_spatial_tokens-1
        z_spatial = z[:self.num_spatial_tokens]  # [num_spatial, batch, d_model]
        z_spatial = self.pos_embed(z_spatial)     # Add 3D PE
        z = torch.cat([z_spatial, z[self.num_spatial_tokens:]], dim=0)  # [seq_len+1, batch, d_model]
        
        # Transformer Encoder: E(Z) = Encoder(Z + PE)
        encoder_output = self.transformer_encoder(z)  # [seq_len+1, batch, d_model]
        
        # === Transformer Decoder (Section 4.2) ===
        
        # Apply 3D PE again on encoder output before decoder (as stated in paper)
        memory = encoder_output.clone()
        memory_spatial = memory[:self.num_spatial_tokens]
        memory_spatial = self.pos_embed(memory_spatial)
        memory = torch.cat([memory_spatial, memory[self.num_spatial_tokens:]], dim=0)
        
        # Prepare decoder queries Q
        tgt = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)  # [max_length, batch, d_model]
        
        # Generate causal mask for parallel decoding
        if tgt_mask is None:
            tgt_mask = self._generate_square_subsequent_mask(self.max_length).to(src.device)
        
        # Transformer Decoder: R = D(Q, E(Z))
        decoder_output = self.transformer_decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
        )  # [max_length, batch, d_model]
        
        # Permute to [batch, max_length, d_model]
        R = decoder_output.permute(1, 0, 2)
        
        # === Spatial Prediction (Section 4.3) ===
        # Ŷ = softmax(FC(R) ⊗ E(Z)^T)
        
        # Project decoder output: FC(R)
        R_proj = self.spatial_fc(R)  # [batch, max_length, d_model]
        
        # Get encoder output for matrix multiplication
        # encoder_output is [seq_len+1, batch, d_model], permute to [batch, seq_len+1, d_model]
        E_Z = encoder_output.permute(1, 0, 2)  # [batch, seq_len+1, d_model]
        
        # Matrix multiplication: FC(R) ⊗ E(Z)^T
        # R_proj: [batch, max_length, d_model]
        # E_Z^T: [batch, d_model, seq_len+1]
        spatial_logits = torch.bmm(R_proj, E_Z.transpose(1, 2))  # [batch, max_length, seq_len+1]
        
        # === Duration Prediction (Section 4.4) ===
        duration_mu = self.duration_mu(R).squeeze(-1)  # [batch, max_length]
        duration_sigma2 = torch.exp(self.duration_logvar(R)).squeeze(-1)  # [batch, max_length]
        
        return {
            'spatial_logits': spatial_logits,
            'duration_mu': duration_mu,
            'duration_sigma2': duration_sigma2,
            'encoder_output': encoder_output,  # For potential auxiliary uses
        }
    
    def _generate_square_subsequent_mask(self, sz: int) -> Tensor:
        """Generate causal mask for parallel decoding"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def inference(
        self,
        src: Tensor,
        temperature: float = 1.0,
        sample: bool = True,
    ):
        """
        Inference mode with sampling for spatial predictions
        
        Args:
            src: Pre-extracted features from Swin UNETR
            temperature: Temperature for softmax sampling
            sample: Whether to sample from distribution or take argmax
            
        Returns:
            Dictionary with predictions including:
            - 'spatial_probs': Softmax probabilities
            - 'predicted_positions': Sampled or argmax positions
            - 'duration_samples': Sampled durations from log-normal
        """
        outputs = self.forward(src)
        
        # Apply temperature scaling and softmax
        spatial_logits = outputs['spatial_logits'] / temperature
        spatial_probs = F.softmax(spatial_logits, dim=-1)
        outputs['spatial_probs'] = spatial_probs
        
        # Sample or argmax positions
        if sample:
            # Reshape for multinomial sampling
            batch_size, seq_len, num_classes = spatial_probs.shape
            probs_flat = spatial_probs.view(-1, num_classes)
            sampled = torch.multinomial(probs_flat, num_samples=1)
            outputs['predicted_positions'] = sampled.view(batch_size, seq_len)
        else:
            outputs['predicted_positions'] = torch.argmax(spatial_probs, dim=-1)
        
        # Sample duration from log-normal distribution
        # t = μ + ε * exp(0.5 * λ), where ε ~ N(0, 1)
        mu = outputs['duration_mu']
        sigma = torch.sqrt(outputs['duration_sigma2'])
        epsilon = torch.randn_like(mu)
        outputs['duration_samples'] = mu + epsilon * sigma
        
        return outputs
