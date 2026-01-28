"""
Foundation Models Benchmark (FMB)

Module: fmb.models.base.trainer
Description: Abstract trainer interface
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from fmb.models.base.config import BaseTrainingConfig
from fmb.models.base.utils import set_seed, setup_amp, format_memory


class BaseTrainer(ABC):
    """
    Abstract base trainer for all FMB models.
    
    Provides standardized training loop with:
    - Automatic mixed precision (AMP)
    - Gradient accumulation
    - Gradient clipping
    - Checkpointing
    - Validation
    - Loss history tracking
    
    Parameters
    ----------
    model : nn.Module
        PyTorch model to train.
    config : BaseTrainingConfig
        Training configuration.
    train_loader : DataLoader
        Training data loader.
    val_loader : Optional[DataLoader]
        Validation data loader (optional).
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: BaseTrainingConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Setup device
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # Setup seed
        set_seed(config.seed)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup AMP
        self.scaler, self.amp_ctx = setup_amp(
            device=config.device,
            dtype=config.amp_dtype,
        )
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        
        # History
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
        }
        
        # Resume if checkpoint provided
        if config.resume_checkpoint:
            self.load_checkpoint(config.resume_checkpoint)
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """
        Create optimizer. Can be overridden by subclasses.
        
        Returns
        -------
        torch.optim.Optimizer
            Configured optimizer.
        """
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
    
    @abstractmethod
    def train_step(self, batch: Any) -> Dict[str, float]:
        """
        Execute one training step.
        
        Must be implemented by subclasses. Should perform forward pass,
        compute loss, and return metrics dictionary.
        
        Parameters
        ----------
        batch : Any
            Batch of training data.
        
        Returns
        -------
        Dict[str, float]
            Dictionary of metrics. Must contain 'loss' key.
        """
        pass
    
    @abstractmethod
    def val_step(self, batch: Any) -> Dict[str, float]:
        """
        Execute one validation step.
        
        Must be implemented by subclasses. Should perform forward pass
        and return metrics dictionary.
        
        Parameters
        ----------
        batch : Any
            Batch of validation data.
        
        Returns
        -------
        Dict[str, float]
            Dictionary of metrics. Must contain 'loss' key.
        """
        pass
    
    def train_epoch(self, epoch: int) -> float:
        """
        Train for one complete epoch.
        
        Parameters
        ----------
        epoch : int
            Current epoch number (1-indexed).
        
        Returns
        -------
        float
            Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.config.epochs}",
            leave=True,
        )
        
        self.optimizer.zero_grad(set_to_none=True)
        
        for step, batch in enumerate(progress):
            # Forward pass with AMP
            with self.amp_ctx:
                metrics = self.train_step(batch)
                loss = metrics["loss"]
                
                # Scale loss for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Optimizer step (with gradient accumulation)
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.grad_clip,
                    )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                
                self.global_step += 1
            
            # Track loss (unscaled)
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1
            
            # Logging
            if (step + 1) % self.config.log_interval == 0:
                current_loss = loss.item() * self.config.gradient_accumulation_steps
                postfix = {"loss": f"{current_loss:.6f}"}
                
                # Add GPU memory if available
                if self.device.type == "cuda":
                    mem_allocated = torch.cuda.memory_allocated()
                    postfix["mem"] = format_memory(mem_allocated)
                
                progress.set_postfix(postfix)
        
        return total_loss / max(num_batches, 1)
    
    @torch.no_grad()
    def validate(self) -> float:
        """
        Validate the model on validation set.
        
        Returns
        -------
        float
            Average validation loss. Returns NaN if no validation loader.
        """
        if self.val_loader is None:
            return float("nan")
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(self.val_loader, desc="Validating", leave=False):
            with self.amp_ctx:
                metrics = self.val_step(batch)
                loss = metrics["loss"]
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def save_checkpoint(
        self,
        epoch: int,
        is_best: bool = False,
        filename: Optional[str] = None,
    ) -> None:
        """
        Save model checkpoint.
        
        Parameters
        ----------
        epoch : int
            Current epoch number.
        is_best : bool
            Whether this is the best checkpoint so far.
        filename : Optional[str]
            Custom filename (default: checkpoint_epoch_{epoch:03d}.pt).
        """
        self.config.out_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "config": self.config.__dict__,
            "history": self.history,
            "best_val_loss": self.best_val_loss,
        }
        
        # Regular checkpoint
        if filename is None:
            filename = f"checkpoint_epoch_{epoch:03d}.pt"
        
        ckpt_path = self.config.out_dir / filename
        torch.save(checkpoint, ckpt_path)
        print(f"💾 Saved checkpoint: {ckpt_path}")
        
        # Best checkpoint
        if is_best:
            best_path = self.config.out_dir / "checkpoint_best.pt"
            torch.save(checkpoint, best_path)
            print(f"New best checkpoint: {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load checkpoint and resume training.
        
        Parameters
        ----------
        checkpoint_path : str
            Path to checkpoint file.
        """
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        
        # Load state
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        
        if "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])
        
        # Restore training state
        self.current_epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.history = checkpoint.get("history", {"train_loss": [], "val_loss": []})
        
        print(f"Resumed from epoch {self.current_epoch}, step {self.global_step}")
    
    def train(self) -> None:
        """
        Execute complete training loop.
        
        Trains for the specified number of epochs, validates after each epoch,
        and saves checkpoints according to the configuration.
        """
        print("=" * 60)
        print(" Starting Training")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Epochs: {self.config.epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"AMP dtype: {self.config.amp_dtype}")
        print("=" * 60)
        
        start_epoch = self.current_epoch + 1
        
        for epoch in range(start_epoch, self.config.epochs + 1):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate()
            
            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            
            # Print summary
            print(f"\n Epoch {epoch} Summary:")
            print(f"   Train Loss: {train_loss:.6f}")
            if not torch.isnan(torch.tensor(val_loss)):
                print(f"   Val Loss:   {val_loss:.6f}")
            
            # Check if best
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                print(f"    New best validation loss!")
            
            # Save checkpoint
            if epoch % self.config.checkpoint_interval == 0:
                self.save_checkpoint(epoch, is_best=is_best)
            
            print()
        
        print("=" * 60)
        print("Training Complete!")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        print("=" * 60)
