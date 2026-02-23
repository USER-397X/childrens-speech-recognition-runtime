import os
import torch
from pathlib import Path
from loguru import logger
from nemo.collections.speechlm2.models import SALM

class CanaryQwenModel:
    def __init__(self, model):
        self.model = model

    @classmethod
    def load(cls, model_dir: Path):
        logger.info(f"Loading Canary-Qwen model offline from {model_dir}")
        
        # Force NeMo to skip Hugging Face hub checks
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        
        # Check GPU availability and assign model to the second GPU if available
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                device = torch.device("cuda:1")
                logger.info("Multiple GPUs detected. Assigning model to the SECOND GPU (cuda:1).")
            else:
                device = torch.device("cuda:0")
                logger.info("Only one GPU detected. Assigning model to the FIRST GPU (cuda:0).")
        else:
            device = torch.device("cpu")
            logger.info("No GPU found. Falling back to CPU.")
        
        # Load the model directly from the local directory
        # We load in bfloat16 to save memory and set it to evaluation mode for inference
        model = SALM.from_pretrained(model_dir).bfloat16().eval().to(device)
        
        logger.info("Model loaded successfully.")
        return cls(model)
    
    def predict_batch(self, audio_paths: list[Path], batch_size: int = 8):
        results = []
        
        # Process in chunks based on batch_size
        for i in range(0, len(audio_paths), batch_size):
            batch_paths = audio_paths[i:i + batch_size]
            
            # NeMo SALM expects prompts as a list of conversations (list of lists)
            prompts = []
            for path in batch_paths:
                conversation = [{
                    "role": "user", 
                    "content": f"Transcribe the following: {self.model.audio_locator_tag}", 
                    "audio": [str(path)]
                }]
                prompts.append(conversation)
                
            # Run the generation
            with torch.inference_mode():
                answer_ids = self.model.generate(
                    prompts=prompts,
                    max_new_tokens=64,
                )
                
            # Decode the token IDs back into text
            for ans in answer_ids:
                # ids_to_text expects CPU tensors
                text = self.model.tokenizer.ids_to_text(ans.cpu())
                results.append(text)
                
        return results