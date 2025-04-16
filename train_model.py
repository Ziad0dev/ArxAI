#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import yaml
import json
import time
import torch
import random
import logging
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Distributed training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# TensorBoard and monitoring
from torch.utils.tensorboard import SummaryWriter
import psutil
import GPUtil

# Set environment variables for HuggingFace libraries
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.embedding_manager import EmbeddingManager
from models.research_model import ResearchModel
from data.dataset import ResearchDataset
from utils.training_utils import (
    set_seed,
    AverageMeter,
    get_optimizer,
    get_scheduler,
    save_checkpoint,
    load_checkpoint,
)

def setup_logging(config):
    """Set up logging configuration"""
    log_dir = Path(config["logging"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"
    
    logging.basicConfig(
        level=getattr(logging, config["logging"]["log_level"].upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def setup_monitoring(config):
    """Set up TensorBoard and system monitoring"""
    tensorboard_dir = Path(config["logging"]["tensorboard_dir"])
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=str(tensorboard_dir / timestamp))
    
    # Initialize Weights & Biases if enabled
    if config["logging"]["wandb_enabled"]:
        try:
            import wandb
            wandb.init(
                project=config["logging"]["wandb_project"],
                name=config["logging"]["wandb_run_name"],
                config=config
            )
            use_wandb = True
        except ImportError:
            logging.warning("Weights & Biases not installed. Continuing without wandb.")
            use_wandb = False
    else:
        use_wandb = False
        
    return writer, use_wandb

def load_data(config, rank=0, world_size=1):
    """Load and prepare datasets"""
    train_dataset = ResearchDataset(
        data_path=config["data"]["train_data_path"],
        embedding_model=config["model"]["embedding_model"],
        max_seq_length=config["data"]["max_seq_length"],
        is_training=True
    )
    
    val_dataset = ResearchDataset(
        data_path=config["data"]["val_data_path"],
        embedding_model=config["model"]["embedding_model"],
        max_seq_length=config["data"]["max_seq_length"],
        is_training=False
    )
    
    # For distributed training
    if world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
    else:
        train_sampler = None
        val_sampler = None
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=(train_sampler is None),
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
        sampler=train_sampler
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
        sampler=val_sampler
    )
    
    return train_loader, val_loader, train_sampler, val_sampler

def create_model(config):
    """Create and initialize the model"""
    embedding_manager = EmbeddingManager(
        embedding_model=config["model"]["embedding_model"],
        embedding_dim=config["model"]["embedding_dim"],
        use_cache=config["system"]["use_cache"],
        cache_dir=config["system"]["cache_dir"]
    )
    
    model = ResearchModel(
        embedding_dim=config["model"]["embedding_dim"],
        hidden_dims=config["model"]["hidden_dims"],
        attention_heads=config["model"]["attention_heads"],
        dropout_rate=config["model"]["dropout_rate"],
        use_domain_adapters=config["model"]["use_domain_adapters"],
        adapter_size=config["model"]["adapter_size"],
        use_citation_graph=config["model"]["use_citation_graph"],
        activation=config["model"]["activation"]
    )
    
    return model, embedding_manager

def validate(model, val_loader, device, criterion, epoch, logger, writer, use_wandb, config, global_step):
    """Validate the model"""
    model.eval()
    val_loss = AverageMeter()
    val_acc = AverageMeter()
    
    with torch.no_grad():
        with tqdm(val_loader, unit="batch", disable=device != 0) as pbar:
            pbar.set_description(f"Validation Epoch {epoch}")
            
            for batch in pbar:
                inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(device)
                
                outputs = model(**inputs)
                loss = criterion(outputs, labels)
                
                # Calculate accuracy
                preds = torch.argmax(outputs, dim=1)
                accuracy = (preds == labels).float().mean()
                
                val_loss.update(loss.item(), labels.size(0))
                val_acc.update(accuracy.item(), labels.size(0))
                
                pbar.set_postfix(loss=val_loss.avg, acc=val_acc.avg)
    
    # Log metrics
    if device == 0:
        logger.info(f"Validation Epoch {epoch}: Loss {val_loss.avg:.4f}, Accuracy: {val_acc.avg:.4f}")
        writer.add_scalar("val/loss", val_loss.avg, global_step)
        writer.add_scalar("val/accuracy", val_acc.avg, global_step)
        
        # Log to wandb if enabled
        if use_wandb:
            import wandb
            wandb.log({
                "val_loss": val_loss.avg,
                "val_accuracy": val_acc.avg,
                "epoch": epoch,
                "global_step": global_step
            })
    
    return val_loss.avg, val_acc.avg

def train_epoch(model, train_loader, optimizer, scheduler, device, criterion, epoch, 
                logger, writer, use_wandb, config, global_step):
    """Train for one epoch"""
    model.train()
    train_loss = AverageMeter()
    train_acc = AverageMeter()
    
    # For gradient accumulation
    accum_steps = config["training"]["grad_accumulation_steps"]
    optimizer.zero_grad()
    
    with tqdm(train_loader, unit="batch", disable=device != 0) as pbar:
        pbar.set_description(f"Training Epoch {epoch}")
        
        for i, batch in enumerate(pbar):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(**inputs)
            loss = criterion(outputs, labels) / accum_steps
            
            # Backward pass with gradient accumulation
            if config["training"]["mixed_precision"]:
                with torch.cuda.amp.autocast():
                    loss.backward()
            else:
                loss.backward()
            
            # Calculate accuracy
            with torch.no_grad():
                preds = torch.argmax(outputs, dim=1)
                accuracy = (preds == labels).float().mean()
                
                train_loss.update(loss.item() * accum_steps, labels.size(0))
                train_acc.update(accuracy.item(), labels.size(0))
            
            # Update weights if accumulated enough gradients
            if (i + 1) % accum_steps == 0 or (i + 1) == len(train_loader):
                # Gradient clipping
                if config["training"]["max_grad_norm"] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        config["training"]["max_grad_norm"]
                    )
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                
                # Log metrics
                if device == 0 and global_step % config["logging"]["log_steps"] == 0:
                    lr = scheduler.get_last_lr()[0]
                    logger.info(
                        f"Epoch {epoch} Step {global_step}: "
                        f"Loss {train_loss.avg:.4f}, Accuracy: {train_acc.avg:.4f}, "
                        f"LR: {lr:.8f}"
                    )
                    
                    writer.add_scalar("train/loss", train_loss.avg, global_step)
                    writer.add_scalar("train/accuracy", train_acc.avg, global_step)
                    writer.add_scalar("train/learning_rate", lr, global_step)
                    
                    # Log system metrics
                    memory = psutil.virtual_memory()
                    writer.add_scalar("system/memory_used_percent", memory.percent, global_step)
                    
                    if torch.cuda.is_available():
                        gpu_util = GPUtil.getGPUs()[device].memoryUtil * 100
                        writer.add_scalar(f"system/gpu{device}_memory_percent", gpu_util, global_step)
                    
                    # Log to wandb if enabled
                    if use_wandb:
                        import wandb
                        wandb.log({
                            "train_loss": train_loss.avg,
                            "train_accuracy": train_acc.avg,
                            "learning_rate": lr,
                            "epoch": epoch,
                            "global_step": global_step,
                            "memory_used_percent": memory.percent
                        })
                
                # Validate and save checkpoint if needed
                if (config["checkpoints"]["evaluation_strategy"] == "steps" and 
                    global_step % config["checkpoints"]["eval_steps"] == 0):
                    val_loss, val_acc = validate(
                        model, val_loader, device, criterion, epoch, 
                        logger, writer, use_wandb, config, global_step
                    )
                    
                    # Save checkpoint
                    if device == 0:
                        save_checkpoint(
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            epoch=epoch,
                            global_step=global_step,
                            val_loss=val_loss,
                            val_acc=val_acc,
                            config=config,
                            is_best=(val_loss < best_val_loss) if "best_val_loss" in locals() else True,
                            filename=f"checkpoint_step_{global_step}.pt"
                        )
                        
                        if "best_val_loss" not in locals() or val_loss < best_val_loss:
                            best_val_loss = val_loss
            
            pbar.set_postfix(loss=train_loss.avg, acc=train_acc.avg)
    
    return global_step, train_loss.avg, train_acc.avg

def train(rank, world_size, config):
    """Main training function for distributed training"""
    # Set up distributed training if needed
    if world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Setup device
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    
    # Set random seed for reproducibility
    set_seed(config["system"]["seed"] + rank)
    
    # Setup logging and monitoring only on rank 0
    if rank == 0:
        logger = setup_logging(config)
        writer, use_wandb = setup_monitoring(config)
        logger.info(f"Starting training with config: {json.dumps(config, indent=2)}")
    else:
        logger = None
        writer = None
        use_wandb = False
    
    # Load data
    train_loader, val_loader, train_sampler, val_sampler = load_data(
        config, rank, world_size
    )
    
    # Create model
    model, embedding_manager = create_model(config)
    model = model.to(device)
    
    # Wrap model with DDP if distributed training
    if world_size > 1:
        model = DDP(model, device_ids=[rank])
    
    # Create optimizer and scheduler
    optimizer = get_optimizer(
        model=model,
        optimizer_name=config["training"]["optimizer"],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )
    
    # Create learning rate scheduler
    scheduler = get_scheduler(
        optimizer=optimizer,
        scheduler_name=config["training"]["scheduler"],
        num_warmup_steps=config["training"]["warmup_steps"],
        num_training_steps=len(train_loader) * config["training"]["epochs"]
    )
    
    # Loss function
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop
    global_step = 0
    best_val_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(1, config["training"]["epochs"] + 1):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        
        # Train for one epoch
        global_step, train_loss, train_acc = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=rank,
            criterion=criterion,
            epoch=epoch,
            logger=logger,
            writer=writer,
            use_wandb=use_wandb,
            config=config,
            global_step=global_step
        )
        
        # Validate at the end of each epoch if evaluation strategy is "epoch"
        if config["checkpoints"]["evaluation_strategy"] == "epoch":
            val_loss, val_acc = validate(
                model=model,
                val_loader=val_loader,
                device=rank,
                criterion=criterion,
                epoch=epoch,
                logger=logger,
                writer=writer,
                use_wandb=use_wandb,
                config=config,
                global_step=global_step
            )
            
            # Save checkpoint on rank 0
            if rank == 0:
                is_best = val_loss < best_val_loss
                best_val_loss = min(val_loss, best_val_loss)
                
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    global_step=global_step,
                    val_loss=val_loss,
                    val_acc=val_acc,
                    config=config,
                    is_best=is_best,
                    filename=f"checkpoint_epoch_{epoch}.pt"
                )
    
    # Clean up
    if rank == 0:
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time/60:.2f} minutes")
        
        if writer:
            writer.close()
        
        if use_wandb:
            import wandb
            wandb.finish()
    
    if world_size > 1:
        dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser(description="Train research model")
    parser.add_argument("--config", type=str, default="config/training_config.yaml",
                      help="Path to configuration file")
    parser.add_argument("--local_rank", type=int, default=-1,
                      help="Local rank for distributed training")
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Create output directories
    os.makedirs(config["checkpoints"]["save_dir"], exist_ok=True)
    os.makedirs(config["logging"]["log_dir"], exist_ok=True)
    os.makedirs(config["logging"]["tensorboard_dir"], exist_ok=True)
    
    # Set CUDA devices if specified
    if "cuda_visible_devices" in config["system"]:
        os.environ["CUDA_VISIBLE_DEVICES"] = config["system"]["cuda_visible_devices"]
    
    # Check for distributed training
    if config["system"]["distributed_training"] and torch.cuda.device_count() > 1:
        world_size = torch.cuda.device_count()
        mp.spawn(train, args=(world_size, config), nprocs=world_size)
    else:
        # Single GPU or CPU training
        train(0, 1, config)

if __name__ == "__main__":
    main() 