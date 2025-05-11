from models.gpt import GPT2Model
from transformers import PreTrainedModel, GPT2Config
from utils.configs import ModelConfig

# hugging face compat
class hfWrapper(PreTrainedModel):
    config_class = GPT2Config
    def __init__(self, config):
        super().__init__(config)
        self.model = GPT2Model(config)
    def forward(self, input_ids):
        return self.model(input_ids)
    
def hfConfig(configs: ModelConfig):
    return GPT2Config(
        vocab_size=configs.vocab_size,
        n_positions=configs.block_size,
        n_ctx=configs.block_size,
        n_embd=configs.n_embd,
        n_layer=configs.n_layers,
        n_head=configs.n_head,
        resid_pdrop=configs.dropout,
        embd_pdrop=configs.dropout,
        attn_pdrop=configs.dropout,
        use_cache=False  # opt
    )
