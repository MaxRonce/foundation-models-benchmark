"""
Foundation Models Benchmark (FMB)

Module: fmb.detection.models
Description: Normalizing Flow architectures for anomaly detection
"""

import torch
import torch.nn as nn
import normflows as nf

def build_coupling_flow(
    dim: int,
    hidden_features: int,
    num_transforms: int,
) -> nn.Module:
    """
    Build a RealNVP-style flow with Affine Coupling Layers.
    Note: Requires even dimensions.
    """
    if dim < 2 or dim % 2 != 0:
        raise RuntimeError(
            f"Embedding dimension {dim} is not supported for RealNVP-style coupling (needs an even dimension >= 2)."
        )
    
    base = nf.distributions.base.DiagGaussian(dim)
    flows: list[nn.Module] = []
    
    # Split dimensions for coupling
    cond_dim = dim // 2
    transformed_dim = dim - cond_dim
    
    for _ in range(num_transforms):
        # MLP for the affine transformation parameters
        net = nf.nets.MLP(
            [cond_dim, hidden_features, hidden_features, transformed_dim * 2],
            init_zeros=True,
        )
        # Affine coupling block
        flows.append(nf.flows.AffineCouplingBlock(net, scale_map="sigmoid"))
        # Permute dimensions
        flows.append(nf.flows.Permute(dim, mode="swap"))
        # ActNorm for training stability
        flows.append(nf.flows.ActNorm((dim,)))
        
    return nf.NormalizingFlow(base, flows)


def build_autoregressive_flow(
    dim: int,
    hidden_features: int,
    num_transforms: int,
) -> nn.Module:
    """
    Build a Masked Autoregressive Flow (MAF).
    More powerful but slower to sample (fast to score density).
    """
    base = nf.distributions.base.DiagGaussian(dim)
    flows: list[nn.Module] = []
    
    for _ in range(num_transforms):
        # Masked Affine Autoregressive Flow
        flows.append(nf.flows.MaskedAffineAutoregressive(features=dim, hidden_features=hidden_features))
        # Permutation to mix dimensions
        flows.append(nf.flows.Permute(dim, mode="swap"))
        # ActNorm
        flows.append(nf.flows.ActNorm(dim))
        
    return nf.NormalizingFlow(base, flows)


def build_flow(
    flow_type: str,
    dim: int,
    hidden_features: int,
    num_transforms: int,
) -> nn.Module:
    """Factory function to build a flow based on type."""
    flow_type = flow_type.lower()
    if flow_type in ("coupling", "realnvp"):
        return build_coupling_flow(dim, hidden_features, num_transforms)
    elif flow_type in ("autoregressive", "maf", "ar"):
        return build_autoregressive_flow(dim, hidden_features, num_transforms)
    else:
        raise ValueError(f"Unknown flow_type: {flow_type}. Options: 'coupling', 'autoregressive'")
