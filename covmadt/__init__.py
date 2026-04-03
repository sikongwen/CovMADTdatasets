"""
CovMADT: Efficient Offline Multi-Agent Reinforcement Learning via Convex Markov Games

This package implements the CovMADT algorithm for offline multi-agent reinforcement learning.
"""

__version__ = "0.1.0"

from .algorithms.covmadt import CovMADT

__all__ = [
    "CovMADT",
]


