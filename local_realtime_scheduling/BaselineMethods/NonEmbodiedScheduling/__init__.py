"""
Non-Embodied Scheduling Baseline Module

This module provides a non-embodied (disembodied) scheduling environment
as a baseline comparison to the embodied scheduling approach.

Key characteristics of the non-embodied approach:
- Event-driven time progression (vs. step-by-step simulation)
- Abstract resource allocation (vs. physical constraints)
- Global/aggregate observations (vs. local views)
- Fixed processing/transport times (vs. dynamic physical effects)
- Logical feasibility checks only (vs. spatial/energy constraints)
"""

from .NonEmbodiedSchedulingMultiAgentEnv import NonEmbodiedSchedulingMultiAgentEnv

__all__ = ["NonEmbodiedSchedulingMultiAgentEnv"]
