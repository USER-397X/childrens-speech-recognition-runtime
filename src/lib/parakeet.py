from pathlib import Path
from loguru import logger
import nemo.collections.asr as nemo_asr
import lightning.pytorch as pl
from omegaconf import DictConfig, OmegaConf, open_dict

class ParakeetModel:
    def __init__(self, model):
        self.model = model

    @classmethod
    def load(cls, model_path):
        logger.info(f"Loading model from: {model_path}")
        model = nemo_asr.models.ASRModel.restore_from(str(model_path)) # ensure it's a string
        return cls(model)
    
    def train(self, train_manifest: Path, val_manifest: Path, epochs: int = 10, batch_size: int = 4):
            logger.info("Setting up training and validation data...")
            
            # 1. Define NeMo data configurations
            train_config = DictConfig({
                'manifest_filepath': str(train_manifest),
                'sample_rate': 16000, 
                'batch_size': batch_size,
                'shuffle': True,
                'num_workers': 4
            })
            
            val_config = DictConfig({
                'manifest_filepath': str(val_manifest),
                'sample_rate': 16000,
                'batch_size': batch_size,
                'shuffle': False,
                'num_workers': 4
            })

            # 2. Hook up the data
            self.model.setup_training_data(train_data_config=train_config)
            self.model.setup_validation_data(val_data_config=val_config)

            # 3. Setup Optimization (CRITICAL FOR FINE-TUNING)
            logger.info("Setting up optimizer...")
            optim_cfg = DictConfig({
                'name': 'adamw',
                'lr': 2e-5,               # Low learning rate for fine-tuning
                'betas': [0.9, 0.98],
                'weight_decay': 1e-3,
                'sched': {
                    'name': 'CosineAnnealing',
                    'warmup_steps': 500,  # Helps prevent catastrophic forgetting
                    'min_lr': 1e-6,
                }
            })
            self.model.setup_optimization(optim_config=optim_cfg)

            #Workaround for CUDA graphs
            # -> ADD THIS WORKAROUND <-
            logger.info("Disabling CUDA graphs for validation decoding...")
            decoding_cfg = self.model.cfg.decoding
            
            # Use open_dict instead of OmegaConf.unlocked
            with open_dict(decoding_cfg):
                decoding_cfg.greedy.use_cuda_graph_decoder = False
                
            self.model.change_decoding_strategy(decoding_cfg)

            # 4. Setup PyTorch Lightning Trainer
            logger.info("Initializing PyTorch Lightning Trainer...")
            trainer = pl.Trainer(
                devices=1, 
                accelerator='gpu', 
                max_epochs=epochs,
                enable_checkpointing=True,
                log_every_n_steps=10,
                
                # --- ADD THESE TWO LINES ---
                precision="16-mixed",       # Uses 16-bit precision to save massive amounts of VRAM
                accumulate_grad_batches=4,  # Updates weights every 4 batches instead of every 1
            )
            
            self.model.set_trainer(trainer)

            # 5. Start Fine-tuning
            logger.info("Starting training loop...")
            trainer.fit(self.model)
            
            # 6. Save the fine-tuned model
            save_path = "parakeet-finetuned.nemo"
            self.model.save_to(save_path)
            logger.success(f"Training complete. Model saved to {save_path}")