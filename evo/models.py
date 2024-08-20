import pkgutil
from transformers import AutoConfig, AutoModelForCausalLM
import yaml

from stripedhyena.utils import dotdict
from stripedhyena.model import StripedHyena
from stripedhyena.tokenizer import CharLevelTokenizer


MODEL_NAMES = ['evo-1-8k-base', 'evo-1-131k-base', 'quant-evo']

class Evo:
    def __init__(self, model_name: str = MODEL_NAMES[1], device: str = None):
        """
        Loads an Evo model checkpoint given a model name.
        If the checkpoint does not exist, we automatically download it from HuggingFace.
        """
        self.device = device

        # Check model name.

        if model_name not in MODEL_NAMES:
            raise ValueError(
                f'Invalid model name {model_name}. Should be one of: '
                f'{", ".join(MODEL_NAMES)}.'
            )

        # Assign config path.

        if model_name in ['evo-1-8k-base', 'quant-evo']:
            config_path = 'configs/evo-1-8k-base_inference.yml'
        elif model_name == 'evo-1-131k-base':
            config_path = 'configs/evo-1-131k-base_inference.yml'
        else:
            raise ValueError(
                f'Invalid model name {model_name}. Should be one of: '
                f'{", ".join(MODEL_NAMES)}.'
            )

        # Load model.

        self.model = load_checkpoint(
            model_name=model_name,
            config_path=config_path,
            device=self.device
        )

        # Load tokenizer.

        self.tokenizer = CharLevelTokenizer(512)


HF_MODEL_NAME_MAP = {
    'evo-1-8k-base': 'togethercomputer/evo-1-8k-base',
    'evo-1-131k-base': 'togethercomputer/evo-1-131k-base',
    'quant-evo': 'monsoon-nlp/quant-evo',
}

def load_checkpoint(
    model_name: str = MODEL_NAMES[1],
    config_path: str = 'evo/configs/evo-1-131k-base_inference.yml',
    device: str = None,
    *args, **kwargs
):
    """
    Load checkpoint from HuggingFace and place it into SH model.
    """

    # Map model name to HuggingFace model name.

    hf_model_name = HF_MODEL_NAME_MAP[model_name]

    # Load model config.
    revision = 'main'
    if model_name != 'quant-evo':
        revision = '1.1_fix'

    model_config = AutoConfig.from_pretrained(
        hf_model_name,
        trust_remote_code=True,
        revision=revision,
    )
    model_config.use_cache = True

    # Load model.

    model = AutoModelForCausalLM.from_pretrained(
        hf_model_name,
        config=model_config,
        trust_remote_code=True,
        revision=revision,
    )

    # Load model state dict & cleanup.

    state_dict = model.backbone.state_dict()
    del model
    del model_config

    # Load SH config.

    config = yaml.safe_load(pkgutil.get_data(__name__, config_path))
    global_config = dotdict(config, Loader=yaml.FullLoader)

    # Load SH Model.

    model = StripedHyena(global_config)
    model.load_state_dict(state_dict, strict=True)
    if 'quant' not in hf_model_name:
        model.to_bfloat16_except_poles_residues()
    if device is not None:
        model = model.to(device)

    return model
