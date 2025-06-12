
from llama_cpp import Llama
import torch
import os
import streamlit as st
import time
import logging

logger = logging.getLogger(__name__)

class ModelHandler:
    def __init__(self, config):
        self.config = config
        self.loaded_models = {}
        self.check_available_models()

    def check_available_models(self):
        self.available_models = []
        model_paths = {
            "Llama 3": self.config.get('llama_model_path'),
            "Mistral": self.config.get('mistral_model_path'),
            "Gemma": self.config.get('gemma_model_path')
        }
        for model_name, path in model_paths.items():
            if path and os.path.exists(path):
                self.available_models.append(model_name)

    @st.cache_resource
    def load_model(_self, model_path):
        try:
            quantization = _self._get_quantization_from_filename(model_path)
            return Llama(
                model_path=model_path,
                n_ctx=_self.config['model_n_ctx'],
                n_batch=_self.config['model_n_batch'],
                n_gpu_layers=-1 if torch.cuda.is_available() else 0,
                f16_kv=True,
                use_mmap=True,
                verbose=False,
                **_self._get_quantization_params(quantization)
            )
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {str(e)}")
            raise


