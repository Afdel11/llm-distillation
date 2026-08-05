from dataclasses import dataclass

import torch


@dataclass
class ConsensusOutput:
    """
    Sortie du module Robust Consensus.
    """

    distribution: torch.Tensor
    consensus: torch.Tensor
    teacher_confidence: torch.Tensor
    teacher_confidences: torch.Tensor


@dataclass
class ARCDOutput:
    """
    Sortie complète du pipeline ARCD.
    """

    loss: torch.Tensor

    lambda_: torch.Tensor

    consensus: torch.Tensor

    teacher_confidence: torch.Tensor

    student_confidence: torch.Tensor

    teacher_distribution: torch.Tensor

    teacher_confidences: torch.Tensor