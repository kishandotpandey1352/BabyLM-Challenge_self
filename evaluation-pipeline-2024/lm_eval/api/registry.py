# Dummy decorator to safely ignore @register_model
def register_model(*args, **kwargs):
    def wrapper(cls):
        return cls
    return wrapper

from lm_eval.models.huggingface import HFLM
from lm_eval.models.dummy import DummyLM
from lm_eval.models.local_completions import LocalCompletionLM
from lm_eval.models.local_chat_completions import LocalChatCompletionLM
from lm_eval.models.anthropic_llms import AnthropicLM, AnthropicChatLM
from lm_eval.models.textsynth import TextSynthLM
from lm_eval.models.gguf import GGUFModel
from lm_eval.models.ggml import GGMLModel
from lm_eval.models.neuron_optimum import OptimumNeuronModel
from lm_eval.models.optimum_lm import OptimumLM
from lm_eval.models.nemo_lm import NemoLM
from lm_eval.models.deepsparse import DeepSparseLM
from lm_eval.models.sparseml import SparseMLLM
from lm_eval.models.vllm_causallms import VLLMCausalLM
from lm_eval.models.mamba_lm import MambaLM
from lm_eval.models.neuralmagic import NeuralMagicLM

MODEL_REGISTRY = {
    "hf": HFLM,
    "huggingface": HFLM,
    "local-completions": LocalCompletionLM,
    "local-chat-completions": LocalChatCompletionLM,
    "anthropic": AnthropicLM,
    "anthropic-chat": AnthropicChatLM,
    "textsynth": TextSynthLM,
    "gguf": GGUFModel,
    "ggml": GGMLModel,
    "neuronx": OptimumNeuronModel,
    "optimum": OptimumLM,
    "nemo_lm": NemoLM,
    "deepsparse": DeepSparseLM,
    "sparseml": SparseMLLM,
    "vllm": VLLMCausalLM,
    "mamba_ssm": MambaLM,
    "neuralmagic": NeuralMagicLM,
    "dummy": DummyLM,
}

def get_model(model_name):
    if model_name == "custom_gpt2":
        from lm_eval.models.custom_gpt2 import CustomGPT2
        return CustomGPT2

    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Attempted to load model '{model_name}', but no model for this name found! "
            f"Supported model names: {', '.join(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name]