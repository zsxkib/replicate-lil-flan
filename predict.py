from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    LogitsWarper,
    LogitsProcessorList,
)
from cog import BasePredictor, Input
from typing import Dict
import torch
import json5
import os

CACHE_DIR = "./src/models"


class BiasLogitsWarper(LogitsWarper):
    """Applies a bias to the logits of specific tokens before softmax.

    This class can be used with the `LogitsProcessorList` in Hugging Face's Transformers
    library to alter the logits produced by a model before softmax is applied during
    text generation.

    The class is not dependent on the `input_ids` as it applies bias to specific token ids
    regardless of the context or sequence of tokens currently being processed.

    Attributes
    ----------
    bias : Dict[int, float]
        A dictionary mapping from token ids to bias values. The bias is added to the
        logits for the corresponding token id. If the bias is -100 or 100, the logit for
        the token id is set to negative or positive infinity, respectively, to essentially
        ban or guarantee the token.

    Methods
    -------
    __call__(input_ids: torch.LongTensor, scores: torch.FloatTensor)
        Applies the bias to the logits. This method is called during the generation process.

    warp_logits_gpu(logits: torch.Tensor) -> torch.Tensor
        The method that actually applies the bias to the logits. Optimized for GPUs.

    warp_logits_cpu(logits: torch.Tensor) -> torch.Tensor
        The method that actually applies the bias to the logits. Optimized for CPUs.

    Example
    -------
    bias_dict = {11859: 8}  # We're using 8 here to heavily bias towards "Greg" (token 11859)
    bias_warper = BiasLogitsWarper(bias_dict)
    logits_processor_list = LogitsProcessorList([bias_warper])
    outputs = model.generate(inputs, logits_processor=logits_processor_list)
    """

    def __init__(self, bias: Dict[int, float]):
        """
        Parameters
        ----------
        bias : Dict[int, float]
            A dictionary mapping from token ids to bias values.
        """

        self.bias = bias

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        """The method called during the generation process.

        Parameters
        ----------
        input_ids : torch.LongTensor
            The input ids for the current generation step. Not used in this class.
        scores : torch.FloatTensor
            The logits for the current generation step.

        Returns
        -------
        torch.Tensor
            The modified logits.
        """

        # input_ids not used because biases are applied to scores (logits) over the entire vocab
        # So, we don't really need input_ids, it's just a formality outlined by the LogitsWarper ABC
        return self.warp_logits_gpu(scores) if torch.cuda.is_available() else self.warp_logits_cpu(scores)

    def warp_logits_gpu(self, logits: torch.Tensor) -> torch.Tensor:
        """Applies the bias to the logits for GPU.

        Parameters
        ----------
        logits : torch.Tensor
            The logits for the current generation step.

        Returns
        -------
        torch.Tensor
            The modified logits.
        """
        # create a tensor for biases
        biases = torch.zeros_like(logits)
        inf_indices, non_inf_indices = [], []

        for token_id, bias_value in self.bias.items():
            if abs(bias_value) == 100:
                # Set logit to extremely high or low value
                new_logit = float("inf") if bias_value > 0 else -float("inf")
                # NOTE: A logit of -∞ (negative infinity) would result in a probability of 0 after applying the softmax function, effectively banning that token.
                # A logit of 0 would result in a non-zero probability after applying the softmax function (the exact value depends on the other logits), so the token could still appear in the output.
                inf_indices.append(token_id)
                biases[..., token_id] = new_logit
            else:
                # Add bias to logit
                non_inf_indices.append(token_id)
                biases[..., token_id] = bias_value

        # for those indices, where bias was not infinite
        logits[..., non_inf_indices] += biases[..., non_inf_indices]
        # for those indices, where bias was infinite
        logits[..., inf_indices] = biases[..., inf_indices]
        return logits

    def warp_logits_cpu(self, logits: torch.Tensor) -> torch.Tensor:
        """Applies the bias to the logits for CPU.

        Parameters
        ----------
        logits : torch.Tensor
            The logits for the current generation step.

        Returns
        -------
        torch.Tensor
            The modified logits.
        """

        for token_id, bias_value in self.bias.items():
            if abs(bias_value) == 100:
                # Set logit to extremely high or low value
                new_logit = float("inf") if bias_value > 0 else -float("inf")
                # NOTE: A logit of -∞ (negative infinity) would result in a probability of 0 after applying the softmax function, effectively banning that token.
                # A logit of 0 would result in a non-zero probability after applying the softmax function (the exact value depends on the other logits), so the token could still appear in the output.
                logits[..., token_id] = new_logit
            else:
                # Add bias to logit
                logits[..., token_id] += bias_value
            new_logit = logits[..., token_id].item()
            # print(f"New logit for token {token_id}: {new_logit}")

        # The '...' (ellipsis) is used here to index into any number of dimensions,
        # For example, if logits is a 3D tensor with shape (batch_size, sequence_length, vocab_size),
        # logits[..., token_id] would be equivalent to logits[:, :, token_id].
        return logits


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""

        os.makedirs(CACHE_DIR, exist_ok=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small", cache_dir=CACHE_DIR)
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small", cache_dir=CACHE_DIR)

    def map_tokens_to_bias(self, bias_dict_str: Dict[str, float]) -> Dict[int, float]:
        """
        Maps a dictionary with strings as keys to a dictionary with corresponding token IDs as keys.
        Unknown words will get mapped to the unknown token id <unk>.

        Parameters
        ----------
        tokenizer
            The tokenizer to use for translating the strings into token IDs.
        bias_dict_str : Dict[str, float]
            A dictionary mapping from strings to bias values.

        Returns
        -------
        Dict[int, float]
            The resulting dictionary mapping from token IDs to bias values.

        Examples
        --------
        >>> bias_dict_str = {"Greg": 8, "Sam": -10}
        >>> bias_dict = map_tokens_to_bias(bias_dict_str) # {11859: 8, 3084: -10}
        """

        bias_dict = {}
        for token, bias in bias_dict_str.items():
            token_ids = self.tokenizer.encode(token, add_special_tokens=False)
            if len(token_ids) > 1:
                raise ValueError(f"The string '{token}' corresponds to more than one token in the tokenizer.")
            bias_dict[token_ids[0]] = bias
        return bias_dict

    def predict(
        self,
        prompt: str = Input(
            description="Prompt for language model",
            min_length=1,
            max_length=512,
        ),
        bias_dict_str: str = Input(
            description="Dictionary mapping from strings to bias values",
            default="{}",
            min_length=0,
            max_length=512,
        ),
        max_output_len: int = Input(
            description="Maximum length of output",
            default=64,
            ge=1,
            le=512,
        ),
    ) -> str:
        """Run a single prediction on the model"""

        inputs = self.tokenizer(prompt, return_tensors="pt").input_ids
        logits_processor_list = None
        if bias_dict_str is not None or bias_dict_str != "{}":
            word_to_bias_dict = json5.loads(bias_dict_str)
            tokenid_to_bias_dict = self.map_tokens_to_bias(word_to_bias_dict)
            bias_warper = BiasLogitsWarper(tokenid_to_bias_dict)
            logits_processor_list = LogitsProcessorList([bias_warper])
        outputs = self.model.generate(
            inputs,
            max_new_tokens=max_output_len,
            logits_processor=logits_processor_list,
        )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
