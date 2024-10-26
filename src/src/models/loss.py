

import torch
import torch.nn as nn
import torch.nn.functional as F

#https://amaarora.github.io/posts/2020-06-29-FocalLoss.html
class FocalLoss(nn.Module):
    def __init__(self, alpha=-1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor,weights:torch.Tensor=None, reduction: str = "mean") -> torch.Tensor:
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

        Args:
            inputs (Tensor): A float tensor of arbitrary shape.
                    The predictions for each example.
            targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                    ``'none'``: No reduction will be applied to the output.
                    ``'mean'``: The output will be averaged.
                    ``'sum'``: The output will be summed. Default: ``'none'``.
        Returns:
            Loss tensor with the reduction option applied.
        """
        # Compute the softmax outputs (probabilities)
        probs = F.sigmoid(inputs)
        if weights is not None:
            assert weights.size() == targets.size(), "Weights must be the same shape as soft_targets"
            weights = weights.to(inputs.device)  # Ensure weights are on the same device as inputs
        
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(),weight=weights, reduction="none")

        #Calculates the probability for the positive class
        p_t = probs * targets + (1 - probs) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = torch.where(targets == 1, self.alpha * (1 - probs) ** self.gamma, (1 - self.alpha) * probs ** self.gamma)
            loss = alpha_t * loss

        # Check reduction option and return loss accordingly
        if reduction == "none":
            pass
        elif reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError(
                f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
            )
        return loss
    
class WeightedMSE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, target, weights=None):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)

        if weights is None:
            weights = torch.ones_like(target)
        else:
            assert weights.size(0) == target.size(0)

        loss = torch.mean(weights * (preds - target) ** 2)
        return loss

class WeightedBCE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, target, weights=None):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        bce = torch.nn.BCEWithLogitsLoss(weight=weights)
        loss = bce(preds, target)
        return loss
    

class R2ccpLoss(nn.Module):
    """
    Conformal Prediction via Regression-as-Classification (Etash Guha et al., 2023).
    Paper: https://neurips.cc/virtual/2023/80610

    :param p: norm of distance measure.
    :param tau: weight of the ‘entropy’ term.
    :param midpoints: the midpoint of each bin.
    """

    def __init__(self, p, tau, midpoints,sigma=0.1):
        super().__init__()
        self.p = p
        self.tau = tau
        self.midpoints = midpoints
        self.sigma = sigma
        self.distance_matrix= self.generate_distance_matrix(midpoints)
    
    def generate_distance_matrix(self,values):
        """
        Generate a distance matrix for a set of continuous or discrete values.

        Args:
        - values (torch.Tensor): Continuous or discrete class values.

        Returns:
        - torch.Tensor: Distance matrix.
        """
        values = values.unsqueeze(1)  # Convert to column vector
        distance_matrix = torch.abs(values - values.T)  # Compute pairwise absolute differences
        return distance_matrix

    def forward(self, preds, target, weights=None):
        """ 
        Compute the cross-entropy loss with regularization

        :param preds: the predictions logits of the model. The shape is batch*K.
        :param target: the truth values. The shape is batch*1.
        :param weights: optional weights for each sample. The shape is batch*1.
        """
        assert not target.requires_grad
        if preds.size(0) != target.size(0):
            raise IndexError(f"Batch size of preds must be equal to the batch size of target.")
        
        target = target.view(-1, 1)
        abs_diff = torch.abs(target - self.midpoints.to(preds.device).unsqueeze(0))

        preds_=torch.nn.functional.softmax(preds, dim=1)

        preds_=preds
        cross_entropy = torch.sum((abs_diff ** self.p) * preds_, dim=1)
        
        penalties = torch.zeros_like(cross_entropy)
        closest_index = torch.argmin(abs_diff, dim=1)
        new_target = torch.zeros(preds.size(0), preds.size(1), device=preds.device)
        new_target[torch.arange(preds.size(0)), closest_index] = 1.0
        self.distance_matrix = self.distance_matrix.to(preds.device)
        penalties = self.distance_matrix[closest_index]
        penalties_values = torch.sum(preds_ * penalties, dim=1)
        losses = cross_entropy + self.tau * penalties_values
        if weights is not None:
            losses = losses * weights
        loss = losses.mean()
        return loss
    
