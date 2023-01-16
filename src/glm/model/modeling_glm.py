
"""GPT-2 model."""

import os
import json
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

import mpu
from model.prompt import PromptSpell
from utils import print_rank_0

PRETRAINED_MODEL_ARCHIVE_MAP = {
    "glm-base": "blocklm-base-blank",
    "glm-large": "blocklm-large-blank",
    "glm-large-multi": "blocklm-large-generation",
    "glm-410m-multi": "blocklm-1.25-generation",
    "glm-515m-multi": "blocklm-1.5-generation",
    "glm-roberta": "blocklm-roberta-large-blank",
    "glm-xxlarge": "blocklm-xxlarge"
}
CONFIG_NAME = 'glm_config.json'
WEIGHTS_NAME = 'mp_rank_00_model_states.pt'


def init_method_normal(std=0.02):
    """Init method based on normal distribution.

    This is only used for embeddings. The transformer has its
    own initializer.
    """

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


class GLMConfig:
    """Configuration class to store the configuration of a `GLMModel`.
    """

    def __init__(self,
                 vocab_size_or_config_json_file,
                 num_layers=24,
                 hidden_size=1024,
                 num_attention_heads=16,
                 embedding_dropout_prob=0.1,
                 attention_dropout_prob=0.1,
                 output_dropout_prob=0.1,
                 max_sequence_length=512,
                 max_memory_length=0,
                 checkpoint_activations=False,
                 checkpoint_num_layers=1,
                 parallel_output=True,
                 relative_encoding=False,
                 block_position_encoding=False,
                 output_predict=True,
                 spell_length=None,
                 spell_func='lstm',
                 attention_scale=1.0, ):
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `GLMModel`.
            hidden_size: Size of the transformer layers.
        """
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.num_layers = num_layers
            self.hidden_size = hidden_size
            self.num_attention_heads = num_attention_heads
            self.embedding_dropout_prob = embedding_dropout_prob
            self.attention_dropout_prob = attention_dropout_prob
            self.output_dropout_prob = output_dropout_prob
            self.max_sequence_length = max_sequence_length
            self.max_memory_length = max_memory_length
            self.checkpoint_activations = checkpoint_activations
            self.checkpoint_num_layers = checkpoint_num_layers
            self.parallel_output = parallel_output
            self.relative_encoding = relative_encoding
            self.block_position_encoding = block_position_encoding
            self.output_predict = output_predict
            self.spell_length = spell_length
            self.spell_func = spell_func
            self.attention_scale = attention_scale
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = GLMConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class PreTrainedGLMModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super(PreTrainedGLMModel, self).__init__()
        if not isinstance(config, GLMConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `GLMConfig`. ".format(
                    self.__class__.__name__))
        self.config = config

    @classmethod
    def from_pretrained(cls, pretrained_model_name, state_dict=None, cache_dir=None, checkpoint_activations=False,
                        checkpoint_num_layers=1, parallel_output=True, output_predict=True, spell_length=None,
                        max_memory_length=None, spell_func='lstm', *inputs, **kwargs):
        """
        Instantiate a PreTrainedBertModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name: either:
                - a str with the name of a pre-trained model to load selected in the list of:
                    . `glm-base`
                    . `glm-large`
                    . `glm-large-multi`
                    . `glm-410m-multi`
                    . `glm-515m-multi`
                    . `glm-roberta`
                    . `glm-xxlarge`
                - a path or url to a pretrained model archive containing:
                    . `glm_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        if pretrained_model_name in PRETRAINED_MODEL_ARCHIVE_MAP:
            checkpoint_name = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name]
        else:
            checkpoint_name = pretrained_model_name
        checkpoint_path = os.path.join(cache_dir, checkpoint_name)
        config_file = os.path.join(checkpoint_path, CONFIG_NAME)
        config = GLMConfig.from_json_file(config_file)
        config.checkpoint_activations = checkpoint_activations
        config.checkpoint_num_layers = checkpoint_num_layers
        config.parallel_output = parallel_output
        config.output_predict = output_predict
        config.spell_func = spell_func
        config.spell_length = spell_length
        if max_memory_length is not None:
            config.max_memory_length = max_memory_length
        print_rank_0("Model config {}".format(config))

        model = cls(config, *inputs, **kwargs)
        if state_dict is None:
            weights_path = os.path.join(checkpoint_path, WEIGHTS_NAME)
            state_dict = torch.load(weights_path)
            state_dict = state_dict['module']
            print(state_dict)


        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if len(missing_keys) > 0:
            print("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            print("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        return model


class GLMModel(PreTrainedGLMModel):
    """GLM Language model.

    The output of the forward method are the logits (parallel or
    serial depending on the `parallel_output` flag.
    """

    def __init__(self, config: GLMConfig):

        super(GLMModel, self).__init__(config)
        self.config = config
        self.parallel_output = config.parallel_output
        self.output_predict = config.output_predict
        self.hidden_size = config.hidden_size

        init_method = init_method_normal(std=0.02)


        self.word_embeddings = mpu.VocabParallelEmbedding(config.vocab_size, config.hidden_size,
                                                          init_method=init_method)


        self.transformer = mpu.GPT2ParallelTransformer(num_layers=config.num_layers,
                                                       hidden_size=config.hidden_size,
                                                       num_attention_heads=config.num_attention_heads,
                                                       max_sequence_length=config.max_sequence_length,
                                                       max_memory_length=config.max_memory_length,
                                                       embedding_dropout_prob=config.embedding_dropout_prob,
                                                       attention_dropout_prob=config.attention_dropout_prob,
                                                       output_dropout_prob=config.output_dropout_prob,
                                                       checkpoint_activations=config.checkpoint_activations,
                                                       checkpoint_num_layers=config.checkpoint_num_layers,
                                                       attention_scale=config.attention_scale,
                                                       relative_encoding=config.relative_encoding,
                                                       block_position_encoding=config.block_position_encoding)
        if config.spell_length is not None:
            self.prompt_spell = PromptSpell(config.spell_length, self.hidden_size, config.spell_func)

    def freeze_transformer(self, tune_prefix_layers=None):
        log_str = "Freeze transformer"
        self.word_embeddings.requires_grad_(False)
        self.transformer.requires_grad_(False)
        if tune_prefix_layers is not None:
            log_str += f" tune {tune_prefix_layers} prefix layers"
            for i in range(tune_prefix_layers):
                self.transformer.layers[i].requires_grad_(True)
        print_rank_0(log_str)

    def forward(self, input_ids, position_ids, attention_mask, *mems, return_memory=False, detach_memory=True,
                prompt_pos=None):

        batch_size = input_ids.size(0)
        words_embeddings = self.word_embeddings(input_ids)
        embeddings = words_embeddings
        if prompt_pos is not None:
            embeddings = embeddings.clone()
            prompt_embeds = self.prompt_spell()
            batch_index = torch.arange(batch_size, device=input_ids.device).unsqueeze(1)
            embeddings[batch_index, prompt_pos] = prompt_embeds

        transformer_output = self.transformer(embeddings, position_ids, attention_mask, mems,
                                              return_memory=return_memory, detach_memory=detach_memory)
        logits, hidden_layers = transformer_output
        outputs = hidden_layers

        if self.output_predict:

            logits_parallel = mpu.copy_to_model_parallel_region(
                logits)
            logits_parallel = F.linear(logits_parallel, self.word_embeddings.weight)

            if self.parallel_output:
                return (logits_parallel, *outputs)

            return (mpu.gather_from_model_parallel_region(logits_parallel), *outputs)
        else:
            return (logits, *outputs)


class EncoderDecoder(torch.nn.Module):
    """Seq2Seq Transformer Model
    The output of the forward method are the logits (parallel or serial depending on the `parallel_output` flag).
    """

    def __init__(self,
                 num_layers,
                 vocab_size,
                 hidden_size,
                 num_attention_heads,
                 embedding_dropout_prob,
                 attention_dropout_prob,
                 output_dropout_prob,
                 max_sequence_length,
                 max_memory_length,
                 checkpoint_activations,
                 checkpoint_num_layers=1,
                 parallel_output=True,
                 output_predict=True
                 ):
        super(EncoderDecoder, self).__init__()

        self.parallel_output = parallel_output
        self.output_predict = output_predict

        init_method = init_method_normal(std=0.02)


        self.word_embeddings = mpu.VocabParallelEmbedding(
            vocab_size, hidden_size, init_method=init_method)


        self.encoder = mpu.GPT2ParallelTransformer(num_layers,
                                                   hidden_size,
                                                   num_attention_heads,
                                                   max_sequence_length,
                                                   max_memory_length,
                                                   embedding_dropout_prob,
                                                   attention_dropout_prob,
                                                   output_dropout_prob,
                                                   checkpoint_activations,
                                                   checkpoint_num_layers)
        self.decoder = mpu.GPT2ParallelTransformer(num_layers,
                                                   hidden_size,
                                                   num_attention_heads,
                                                   max_sequence_length,
                                                   max_memory_length,
                                                   embedding_dropout_prob,
                                                   attention_dropout_prob,
                                                   output_dropout_prob,
                                                   checkpoint_activations,
                                                   checkpoint_num_layers,
                                                   use_decoder_layer=True)

    def forward(self, source_ids, target_ids, source_position_ids, target_position_ids, source_mask, target_mask):

        source_embeddings = self.word_embeddings(source_ids)
        target_embeddings = self.word_embeddings(target_ids)


        encoder_output, _ = self.encoder(source_embeddings, source_position_ids, source_mask)
        decoder_output, _ = self.decoder(target_embeddings, target_position_ids, target_mask)
        if self.output_predict:

            output_parallel = mpu.copy_to_model_parallel_region(decoder_output)
            logits_parallel = F.linear(output_parallel, self.word_embeddings.weight)

            if self.parallel_output:
                return (logits_parallel,)

            return (mpu.gather_from_model_parallel_region(logits_parallel),)
        else:
            return (decoder_output,)


def glm_get_params_for_weight_decay_optimization(module):
    weight_decay_params = {'params': []}
    no_weight_decay_params = {'params': [], 'weight_decay': 0.0}
    for module_ in module.modules():
        if isinstance(module_, (mpu.LayerNorm, torch.nn.LayerNorm)):
            no_weight_decay_params['params'].extend(
                [p for p in list(module_._parameters.values())
                 if p is not None and p.requires_grad])
        else:
            weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                 if p is not None and p.requires_grad and n != 'bias'])
            no_weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                 if p is not None and p.requires_grad and n == 'bias'])

    return weight_decay_params, no_weight_decay_params
