import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List

@dataclass

class ModelConfig:
    d_model: int = 32
    n_layers: int = 2
    n_heads: int = 2
    d_ff: int = 64
    vocab_size: int = 17
    max_seq_len: int = 10



class TinyAttentionHead(nn.Module):
    """single attention head with instrumentation"""
    def __init__(self, d_model, head_dim):
        super().__init__()
        self.head_dim = head_dim
        self.q_proj = nn.Linear(d_model, head_dim)
        self.k_proj = nn.Linear(d_model, head_dim)
        self.v_proj = nn.Linear(d_model, head_dim)
        
        # instrumentation
        self.attention_patterns = []
        self.activation_magnitudes = []
        
    def forward(self, x, return_metadata=False):
        batch, seq, _ = x.shape
        
        Q = self.q_proj(x)  # (batch, seq, head_dim)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = torch.softmax(scores, dim=-1)
        
        # track attention patterns
        if return_metadata:
            self.attention_patterns.append(attn_weights.detach())
            self.activation_magnitudes.append(torch.norm(V, dim=-1).detach())
        
        output = torch.matmul(attn_weights, V)
        
        if return_metadata:
            return output, {
                'attn_weights': attn_weights,
                'q_norm': torch.norm(Q, dim=-1),
                'k_norm': torch.norm(K, dim=-1),
                'v_norm': torch.norm(V, dim=-1),
                'attn_entropy': self._compute_entropy(attn_weights)
            }
        
        return output
    
    def _compute_entropy(self, probs):
        # attention entropy - high = confused/uncertain
        return -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)




class InstrumentedTransformer(nn.Module):
    """tiny transformer with extensive instrumentation"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_encoding = nn.Parameter(torch.randn(config.max_seq_len, config.d_model))
        
        # FIX: actually create the layers
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.n_layers)
        ])
        
        self.output = nn.Linear(config.d_model, config.vocab_size)
        
        # instrumentation storage
        self.layer_activations = []
        self.gradient_flows = []
        self.loss_history = []
    
    def forward(self, x, return_metadata=False):
        # embed
        x = self.embedding(x) + self.pos_encoding[:x.size(1)]
        
        metadata = {'layers': []}
        
        # Layers
        for i, layer in enumerate(self.layers):
            if return_metadata:
                x, layer_meta = layer(x, return_metadata=True)
                metadata['layers'].append(layer_meta)
            else:
                # FIX: layers(x) -> layer(x)
                x = layer(x)
        
        # Output (FIX: self.outputs -> self.output)
        logits = self.output(x)
        
        if return_metadata:
            return logits, metadata
        return logits


class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([
            TinyAttentionHead(config.d_model, config.d_model // config.n_heads)
            for _ in range(config.n_heads)
        ])
        self.head_projection = nn.Linear(config.d_model, config.d_model)
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.ReLU(),
            nn.Linear(config.d_ff, config.d_model)
        )
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)

    def forward(self, x, return_metadata=False):
        # multi head attention
        if return_metadata:
            head_outputs = []
            head_metadata = []
            for head in self.heads:
                out, meta = head(x, return_metadata=True)
                head_outputs.append(out)
                head_metadata.append(meta)

            attn_out = torch.cat(head_outputs, dim=-1)
            attn_out = self.head_projection(attn_out)
            x = self.norm1(x + attn_out)

            ff_out = self.ff(x)
            x = self.norm2(x + ff_out)

            return x, {
                'heads': head_metadata,
                'ff_activation': torch.norm(ff_out, dim=-1),
                'residual_norm': torch.norm(x, dim=-1)
            }
        else:
            # standard forward
            head_outputs = [head(x) for head in self.heads]
            attn_out = torch.cat(head_outputs, dim=-1)
            attn_out = self.head_projection(attn_out)
            x = self.norm1(x + attn_out)
            x = self.norm2(x + self.ff(x))
            return x



class ConfusionDetector:
    '''test different confusion metrics'''

    def __init__(self):
        self.metrics_history = {
            'loss': [],
            'attention_entropy': [],
            'gradient_variance': [],
            'layer_disagreement': [],
            'activation_magnitude': [],
            'prediction_confidence': []
        }

    def compute_all_metrics(self, model, batch, metadata, loss):
        # compute every possible confusion signal

        metrics = {}
        
        # Metric 1: Raw Loss
        metrics['loss'] = loss.item()
        
        # Metric 2: Attention Entropy
        entropies = []
        for layer_meta in metadata['layers']:
            for head_meta in layer_meta['heads']:
                entropies.append(head_meta['attn_entropy'].mean().item())
        metrics['attention_entropy'] = np.mean(entropies) if entropies else 0.0
        
        # Metric 3: Gradient Variance (only if training)
        if model.training:
            loss_for_grad = loss.clone()
            loss_for_grad.backward(retain_graph=True)
            grad_norms = []
            for param in model.parameters():
                if param.grad is not None:
                    grad_norms.append(param.grad.norm().item())
            metrics['gradient_variance'] = np.std(grad_norms) if grad_norms else 0.0
            model.zero_grad()
        else:
            metrics['gradient_variance'] = 0.0
        
        # Metric 4: Layer Disagreement
        layer_norms = [layer_meta['residual_norm'].mean().item() 
                       for layer_meta in metadata['layers']]
        metrics['layer_disagreement'] = np.std(layer_norms) if layer_norms else 0.0
        
        # Metric 5: Activation Magnitude
        activations = []
        for layer_meta in metadata['layers']:
            activations.append(layer_meta['ff_activation'].mean().item())
        metrics['activation_magnitude'] = np.mean(activations) if activations else 0.0
        
        # Metric 6: Prediction Confidence
        logits = model(batch['input'], return_metadata=False)
        with torch.no_grad():
            probs = torch.softmax(logits, dim=-1)
            max_probs = probs.max(dim=-1)[0]
            metrics['prediction_confidence'] = max_probs.mean().item()
        
        return metrics

    def track_over_time(self, metrics):
        # store metrics for temporal analysis
        for key, value in metrics.items():
            self.metrics_history[key].append(value)