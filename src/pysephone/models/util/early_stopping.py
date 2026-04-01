"""
Early stopping utility for PyTorch model training.

This module provides an EarlyStopper class that monitors validation loss
and stops training when the model stops improving, preventing overfitting.

Reference:
    https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
"""

from typing import List, Optional, Dict, Any


class EarlyStopper:
    """
    Early stopping criterion for model training.
    
    Monitors validation loss and triggers early stopping when the loss
    doesn't improve for a specified number of epochs (patience). An
    improvement is defined as a decrease in loss by at least min_delta.
    
    Attributes:
        patience: Number of epochs to wait before stopping.
        min_delta: Minimum change in loss to qualify as improvement.
        counter: Current number of epochs without improvement.
        min_validation_loss: Best (lowest) validation loss seen so far.
        best_epoch: Epoch number where best loss was achieved.
        loss_history: List of all validation losses seen.
    
    Example:
        >>> stopper = EarlyStopper(patience=5, min_delta=0.001)
        >>> for epoch in range(num_epochs):
        ...     val_loss = validate(model, val_loader)
        ...     if stopper.early_stop(val_loss):
        ...         print(f"Early stopping at epoch {epoch}")
        ...         break
    """
    
    def __init__(self, patience: int = 1, min_delta: float = 0.0) -> None:
        """
        Initialize the EarlyStopper.
        
        Args:
            patience: Number of epochs to wait without improvement before
                    stopping. Must be >= 1.
            min_delta: Minimum change in validation loss to qualify as an
                      improvement. Must be >= 0.
        
        Raises:
            ValueError: If patience < 1 or min_delta < 0.
        """
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        if min_delta < 0:
            raise ValueError(f"min_delta must be >= 0, got {min_delta}")
        
        self.patience: int = patience
        self.min_delta: float = min_delta
        self.counter: int = 0
        self.min_validation_loss: float = float('inf')
        self.best_epoch: Optional[int] = None
        self.loss_history: List[float] = []
    
    def _is_worse(self, validation_loss: float) -> bool:
        """
        Check if the current loss is worse than the best loss.
        
        A loss is considered worse if it's greater than the best loss
        plus the minimum delta threshold.
        
        Args:
            validation_loss: Current validation loss.
        
        Returns:
            True if the loss is worse (no improvement), False otherwise.
        """
        return validation_loss > (self.min_validation_loss + self.min_delta)
    
    def early_stop(self, validation_loss: float, epoch: Optional[int] = None) -> bool:
        """
        Check if training should stop early.
        
        Updates the best loss if improvement is found, or increments
        the counter if no improvement. Returns True if patience is exceeded.
        
        Args:
            validation_loss: Current validation loss value.
            epoch: Optional epoch number for tracking best epoch.
        
        Returns:
            True if training should stop (patience exceeded), False otherwise.
        """
        self.loss_history.append(validation_loss)
        
        if validation_loss < self.min_validation_loss:
            # Improvement found
            self.min_validation_loss = validation_loss
            self.counter = 0
            if epoch is not None:
                self.best_epoch = epoch
            return False
        elif self._is_worse(validation_loss):
            # No improvement - increment counter
            self.counter += 1
            if self.counter >= self.patience:
                return True
        # Loss is equal or within min_delta - don't increment counter
        # This allows for small fluctuations without triggering early stop
        
        return False
    
    def reset(self) -> None:
        """
        Reset the early stopper to initial state.
        
        Useful for reusing the same EarlyStopper instance for multiple
        training runs.
        """
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.best_epoch = None
        self.loss_history = []
    
    def improved(self) -> bool:
        """
        Check if the last call to early_stop() found an improvement.
        
        Returns:
            True if the last validation loss was better than the previous best.
        """
        return self.counter == 0 and len(self.loss_history) > 0
    
    def get_best_loss(self) -> Optional[float]:
        """
        Get the best validation loss seen so far.
        
        Returns:
            Best validation loss, or None if no losses have been recorded.
        """
        if self.min_validation_loss == float('inf'):
            return None
        return self.min_validation_loss
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the early stopper.
        
        Returns:
            Dictionary containing all state information:
            - patience: Patience setting
            - min_delta: Minimum delta setting
            - counter: Current counter value
            - min_validation_loss: Best loss
            - best_epoch: Best epoch number
            - loss_history: List of all losses
        """
        return {
            'patience': self.patience,
            'min_delta': self.min_delta,
            'counter': self.counter,
            'min_validation_loss': self.min_validation_loss,
            'best_epoch': self.best_epoch,
            'loss_history': self.loss_history.copy(),
        }
