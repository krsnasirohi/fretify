import torch
import torch.nn as nn
import torch.nn.functional as F

class GuitarFoundationalModel(nn.Module):
    def __init__(self, input_bins=252, latent_dim=512, num_pitches=49, num_frets=23):
        super().__init__()
        # 1. Shared Backbone Encoder
        self.backbone = nn.Sequential(
            nn.Linear(input_bins, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.ReLU()
        )
        
        # 2. Head A: Acoustic Pitch Head
        self.pitch_head = nn.Linear(latent_dim, num_pitches)
        
        # 3. Head B: Tablature Fretboard Head
        self.tab_head = nn.Linear(latent_dim, 6 * num_frets)
        self.num_frets = num_frets
        self.num_pitches = num_pitches

    def forward(self, x):
        # x shape: (batch, time_frames, input_bins)
        batch_size, time_frames, _ = x.shape
        
        # Pass through shared encoder
        latent = self.backbone(x)
        
        # Head A processing (probabilities per pitch class)
        pitch_logits = self.pitch_head(latent)
        pitch_probs = torch.sigmoid(pitch_logits)
        
        # Head B processing (probabilities per string per fret class)
        tab_logits = self.tab_head(latent)
        tab_logits = tab_logits.view(batch_size, time_frames, 6, self.num_frets)
        tab_probs = F.softmax(tab_logits, dim=-1)
        
        return pitch_probs, tab_probs