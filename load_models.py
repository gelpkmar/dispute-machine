import sys
import logging
import torch

from huggingface_hub import hf_hub_download
from langchain.llms import LlamaCpp
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig


def load_quantized_model_gguf_ggml(model_id, model_basename, device_type, logging):
    """
    Load a GGUF/GGML quantized model using LlamaCpp.
    """
    try:
        logging.info("Using Llamacpp for GGUF/GGML quantized models")
        model_path = hf_hub_download(
            repo_id=model_id,
            filename=model_basename,
            resume_download=True,
            cache_dir=MODELS_PATH,
        )
        kwargs = {
            "model_path": model_path,
            "n_ctx": CONTEXT_WINDOW_SIZE,
            "max_tokens": MAX_NEW_TOKENS,
            "n_batch": N_BATCH,
        }
        if device_type.lower() == "mps":
            kwargs["n_gpu_layers"] = 1
        if device_type.lower() == "cuda":
            kwargs["n_gpu_layers"] = N_GPU_LAYERS

        return LlamaCpp(**kwargs)
    except TypeError:
        if "ggml" in model_basename:
            logging.info("If you were using GGML model, LLAMA-CPP Dropped Support, Use GGUF Instead")
        return None


def load_quantized_model_qptq(model_id, model_basename, device_type, logging):
    """
    This function is disabled since it requires auto_gptq
    """
    logging.warning("GPTQ quantized models are not supported in this version. Please use GGUF models instead.")
    return None, None


def load_full_model(model_id, model_basename, device_type, logging):
    """
    Load a full model using either LlamaTokenizer or AutoModelForCausalLM.
    """
    if device_type.lower() in ["mps", "cpu", "hpu"]:
        logging.info("Using AutoModelForCausalLM")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir="./models/"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="./models/")
    else:
        logging.info("Using AutoModelForCausalLM for full models")
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="./models/")
        logging.info("Tokenizer loaded")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            llm_int8_enable_fp32_cpu_offload=True  # ✅ Enables CPU offloading
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            cache_dir=MODELS_PATH,
            trust_remote_code=True,
            quantization_config=bnb_config
        )
        model.tie_weights()
    return model, tokenizer


def load_quantized_model_awq(model_id, logging):
    """
    This function is disabled since it requires special dependencies
    """
    logging.warning("AWQ quantized models are not supported in this version.")
    return None, None