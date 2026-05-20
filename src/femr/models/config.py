from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import transformers


class FEMRTransformerConfig(transformers.PretrainedConfig):
    def __init__(
        self,
        vocab_size: int = 32768,
        is_hierarchical: bool = False,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        n_heads: int = 12,
        n_layers: int = 6,
        attention_width: int = 496,
        use_normed_ages: bool = False,
        use_bias: bool = True,
        hidden_act: str = "gelu",
        use_reasoning_layer: bool = False,
        reasoning_top_k: int = 32,
        reasoning_weight: float = 1.0,
        reasoning_embedding_init_path: Optional[str] = None,
        reasoning_constrain_to_history: bool = False,
        **kwargs,
    ) -> None:
        """Defined a configuration for a FEMR Transformer.

        Arguments:
            vocab_size: The number of tokens in the vocabulary
            is_hierarchical: Whether to use a hierarchical vocabulary. See FEMRTokenizer for more information
            hidden_size: The internal representation size
            intermediate_size: The size of the FFN in the transformer layers
            n_heads: The number of attention heads
            n_layers: The number of transformer encoder layers
            attention_width: FEMR by default uses a local attention transformer with a width defined here
            use_normed_ages: Whether or not to provide normalized ages as a feature to the model
            use_bias: Whether or not to use bias terms in the transformer layers
            hidden_act: The type of activation function to use in the transformer
            use_reasoning_layer: Whether to insert a vocab-attention reasoning layer before task heads
            reasoning_top_k: Number of top vocab tokens to attend over in the reasoning layer
            reasoning_weight: Mixing weight alpha; output = alpha*reasoning + (1-alpha)*hidden_state
            reasoning_embedding_init_path: Optional location of a (vocab_size, hidden_size)
                torch tensor used to initialize the reasoning_embedding weight. Three forms
                are accepted:
                  * Local filesystem path.
                  * hf://<repo_id>/<filename>, e.g.,
                    hf://trialspark/femr-reasoning-init/reasoning_init.pt
                  * Full https URL like https://huggingface.co/<repo>/resolve/<rev>/<file>
                The init only takes effect when the model is constructed from scratch; when
                loading from a trained checkpoint, the saved state_dict overrides it.
            reasoning_constrain_to_history: If True, the reasoning layer's top-k selection
                is restricted to vocab tokens that appear in the *same patient's* tokenized
                history at or before each prediction position. Honors MOTOR's sample-packing
                convention (multiple patients' tokens interleaved in one flat sequence) by
                using the per-position segment IDs derived from subject_ids — so a prediction
                at position p cannot attend to tokens that belong to a different patient
                packed later in the same batch, nor to future tokens within the same patient.
                Vocab logits for out-of-history tokens are masked to -inf before topk.
        """
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.is_hierarchical = is_hierarchical

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.attention_width = attention_width

        self.use_normed_ages = use_normed_ages

        self.use_bias = use_bias
        self.hidden_act = hidden_act

        self.use_reasoning_layer = use_reasoning_layer
        self.reasoning_top_k = reasoning_top_k
        self.reasoning_weight = reasoning_weight
        self.reasoning_embedding_init_path = reasoning_embedding_init_path
        self.reasoning_constrain_to_history = reasoning_constrain_to_history


class FEMRTaskConfig(transformers.PretrainedConfig):
    def __init__(self, task_type: str = "", task_kwargs: Mapping[str, Any] = {}, **kwargs):
        """A generic FEMR task definition. This holds state used for initalizing a tasks.py class.

        Task.get_task_config returns the task type and kwargs used to initialize this.

        Arguments:
            task_type: The name of the task.
            task_kwargs: Arbitrary arguments used to store state for that task.
        """
        super().__init__(**kwargs)
        self.task_type = task_type
        self.task_kwargs = task_kwargs


class FEMRModelConfig(transformers.PretrainedConfig):
    """A model config is defined as the combination of a transformer config and a task config."""

    def __init__(
        self,
        transformer_config: Optional[Dict[str, Any]] = None,
        task_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """A combination of a transformer config and a task config.

        It is possible to initialize this with only a transformer config, in which
        case the model will be configured for inference only.
        """
        super().__init__(**kwargs)
        if transformer_config is None:
            transformer_config = {}
        self.transformer_config = FEMRTransformerConfig(**transformer_config)

        self.task_config: Optional[FEMRTaskConfig]

        if task_config is not None:
            self.task_config = FEMRTaskConfig(**task_config)
        else:
            self.task_config = None

    @classmethod
    def from_transformer_task_configs(
        cls, transformer_config: FEMRTransformerConfig, task_config: FEMRTaskConfig
    ) -> FEMRModelConfig:
        """
        Combine a transformer configuration and task configuration into a model configuration.
        """
        if task_config is not None:
            task_config_dict = task_config.to_dict()
        else:
            task_config_dict = None

        return cls(transformer_config=transformer_config.to_dict(), task_config=task_config_dict)
