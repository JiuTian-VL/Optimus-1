# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from dataclasses import dataclass

import torch
from PIL.Image import Image
from transformers import LlamaTokenizerFast
from transformers.processing_utils import ProcessorMixin

from ..utils.conversation import get_conv_template

from .image_processing_vlm import VLMImageProcessor


class DictOutput:
    def keys(self):
        return self.__dict__.keys()

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, key, value):
        self.__dict__[key] = value


@dataclass
class VLChatProcessorOutput(DictOutput):
    sft_format: str
    input_ids: torch.Tensor
    pixel_values: torch.Tensor
    num_image_tokens: torch.IntTensor

    def __len__(self):
        return len(self.input_ids)


@dataclass
class VLChatProcessorTrainOutput(VLChatProcessorOutput):
    labels: torch.Tensor  # answer
    history: dict | None = None

    def __len__(self):
        return len(self.input_ids)


@dataclass
class VLChatProcessorDPOTrainOutput(DictOutput):
    prompt: str
    chosen_input_ids: torch.Tensor
    rejected_input_ids: torch.Tensor
    chosen_labels: torch.Tensor
    rejected_labels: torch.Tensor
    pixel_values: torch.Tensor

    def __len__(self):
        return len(self.chosen_input_ids)


@dataclass
class BatchedVLChatProcessorOutput(DictOutput):
    sft_format: list[str]
    input_ids: torch.Tensor
    pixel_values: torch.Tensor
    attention_mask: torch.Tensor
    images_seq_mask: torch.BoolTensor
    images_emb_mask: torch.BoolTensor

    def to(self, device, dtype=torch.bfloat16):
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        self.images_seq_mask = self.images_seq_mask.to(device)
        self.images_emb_mask = self.images_emb_mask.to(device)
        self.pixel_values = self.pixel_values.to(device=device, dtype=dtype)
        return self

    def __len__(self):
        return self.input_ids.size(0)


@dataclass
class BatchedVLChatProcessorTrainOutput(BatchedVLChatProcessorOutput):
    labels: torch.Tensor
    history: dict | None = None

    def to(self, device, dtype=torch.bfloat16):
        super().to(device, dtype)
        self.labels = self.labels.to(device=device)
        return self


class VLChatProcessor(ProcessorMixin):
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = ("LlamaTokenizer", "LlamaTokenizerFast")

    attributes = ["image_processor", "tokenizer"]

    system_prompt = (
        "You are a helpful language and vision assistant. "
        "You are able to understand the visual content that the user provides, "
        "and assist the user with a variety of tasks using natural language."
    )

    def __init__(
        self,
        image_processor: VLMImageProcessor,
        tokenizer: LlamaTokenizerFast,
        image_tag: str = "<image_placeholder>",
        num_image_tokens: int = 576,
        add_special_token: bool = False,
        sft_format: str = "deepseek",
        mask_prompt: bool = True,
        ignore_id: int = -100,
        is_train: bool = True,
        **kwargs,
    ):
        self.image_processor = image_processor
        self.tokenizer = tokenizer

        image_id = self.tokenizer.vocab.get(image_tag)
        if image_id is None:
            special_tokens = [image_tag]
            special_tokens_dict = {"additional_special_tokens": special_tokens}
            self.tokenizer.add_special_tokens(special_tokens_dict)
            print(f"Add image tag = {image_tag} to the tokenizer")

        self.image_tag = image_tag
        self.num_image_tokens = num_image_tokens
        self.add_special_token = add_special_token
        self.sft_format = sft_format
        self.mask_prompt = mask_prompt
        self.ignore_id = ignore_id

        self.is_train = is_train

        self.chat_template = get_conv_template(sft_format)

        super().__init__(
            image_processor,
            tokenizer,
            image_tag,
            num_image_tokens,
            add_special_token,
            sft_format,
            mask_prompt,
            ignore_id,
            **kwargs,
        )

    def new_chat_template(self):
        conv = get_conv_template(self.sft_format)
        conv.set_system_message(self.system_prompt)
        return conv

    def make_single_turn_conv(
        self, prompt: str, response: str, image_list: list[str] | None = None
    ):
        return [
            {
                "role": "User",
                "content": prompt,
                "images": image_list if image_list is not None else [],
            },
            {"role": "Assistant", "content": response},
        ]

    def process_batch_conv(
        self,
        conversations: list[list[dict[str, str]]],
        system_message: str | None = None,
        add_end_for_empty_value=False,
    ):
        conv_template = get_conv_template(self.sft_format)
        role_begin = {
            "User": conv_template.roles[0],
            "Assistant": conv_template.roles[1],
        }
        role_end = {
            "User": conv_template.sep,
            "Assistant": conv_template.sep2,
        }
        raw_texts = []
        batch_input_ids = []
        batch_attention_masks = []
        batch_labels = []
        for source in conversations:
            __raw_text = ""
            attention_masks = []
            labels = []
            previous_len = 0
            for idx, sentence in enumerate(source):
                begin = role_begin[sentence["role"]]
                end = role_end[sentence["role"]]
                extend_text = (
                    begin
                    + sentence["content"]
                    + (
                        end
                        if sentence["content"] != "" or add_end_for_empty_value
                        else ""
                    )
                )
                __raw_text += extend_text
                text_tokens = self.tokenizer(
                    sentence["content"], padding=False, add_special_tokens=(idx == 0)
                )
                current_tokens = self.tokenizer(__raw_text)
                input_ids = current_tokens["input_ids"]
                attention_masks = current_tokens["attention_mask"]
                extend_len = len(input_ids) - previous_len
                previous_len = len(input_ids)
                labels.extend([-100] * extend_len)
                if (
                    sentence["role"] == "Assistant"
                    and len(text_tokens["input_ids"]) != 0
                ):
                    target_len = min(
                        [extend_len, len(text_tokens["input_ids"]), len(labels)]
                    )
                    labels[-target_len:] = text_tokens["input_ids"][-target_len:]

            labels = [
                label if mask == 1 else -100
                for label, mask in zip(labels, attention_masks)
            ]
            assert (
                len(input_ids) == len(attention_masks) == len(labels)
            ), f"input_ids:{len(input_ids)}, attention_masks:{len(attention_masks)}, labels:{len(labels)}"
            batch_input_ids.append(input_ids)
            batch_attention_masks.append(attention_masks)
            batch_labels.append(labels)
            raw_texts.append(__raw_text)
        return {
            "prompt": None,
            "answer": None,
            "full": dict(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_masks,
                labels=batch_labels,
            ),
            "raw_str": raw_texts,
        }

    def apply_sft_template_for_multi_turn_prompts(
        self,
        conversations: list[dict[str, str]],
        sft_format: str = "deepseek",
        system_prompt: str = "",
    ):
        """
        Applies the SFT template to conversation.

        An example of conversation:
        conversation = [
            {
                "role": "User",
                "content": "<image_placeholder> is Figure 1.\n<image_placeholder> is Figure 2.\nWhich image is brighter?",
                "images": [
                    "./multi-images/attribute_comparison_1.png",
                    "./multi-images/attribute_comparison_2.png"
                ]
            },
            {
                "role": "Assistant",
                "content": ""
            }
        ]

        Args:
            conversations (List[Dict]): A conversation with a List of Dict[str, str] text.
            sft_format (str, optional): The format of the SFT template to use. Defaults to "deepseek".
            system_prompt (str, optional): The system prompt to use in the SFT template. Defaults to "".

        Returns:
            sft_prompt (str): The formatted text.
        """

        conv = get_conv_template(sft_format)
        conv.set_system_message(system_prompt)
        for message in conversations:
            conv.append_message(message["role"], message["content"].strip())
        sft_prompt = conv.get_prompt().strip()

        return sft_prompt

    @property
    def image_token(self):
        return self.image_tag

    @property
    def image_id(self):
        image_id = self.tokenizer.vocab.get(self.image_tag)
        return image_id

    @property
    def pad_id(self):
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        assert pad_id is not None
        return pad_id

    def add_image_token(
        self,
        image_indices: list[int],
        input_ids: torch.LongTensor,
    ):
        """

        Args:
            image_indices (List[int]): [index_0, index_1, ..., index_j]
            input_ids (torch.LongTensor): [N]

        Returns:
            input_ids (torch.LongTensor): [N + image tokens]
            num_image_tokens (torch.IntTensor): [n_images]
        """

        input_slices = []

        start = 0
        for index in image_indices:
            if self.add_special_token:
                end = index + 1
            else:
                end = index

            # original text tokens
            input_slices.append(input_ids[start:end])

            # add image tokens, and set the mask as False
            input_slices.append(
                self.image_id * torch.ones((self.num_image_tokens,), dtype=torch.long)
            )
            start = index + 1

        # the left part
        input_slices.append(input_ids[start:])

        # concat all slices
        input_ids = torch.cat(input_slices, dim=0)
        num_image_tokens = torch.IntTensor([self.num_image_tokens] * len(image_indices))

        return input_ids, num_image_tokens

    def process_one(
        self,
        prompt: str = None,
        conversations: list[dict[str, str]] = None,
        images: list[Image] = None,
        **kwargs,
    ):
        """

        Args:
            prompt (str): the formatted prompt;
            conversations (List[Dict]): conversations with a list of messages;
            images (List[ImageType]): the list of images;
            **kwargs:

        Returns:
            outputs (BaseProcessorOutput): the output of the processor,
                - input_ids (torch.LongTensor): [N + image tokens]
                - target_ids (torch.LongTensor): [N + image tokens]
                - images (torch.FloatTensor): [n_images, 3, H, W]
                - image_id (int): the id of the image token
                - num_image_tokens (List[int]): the number of image tokens
        """

        assert (
            prompt is None or conversations is None
        ), "prompt and conversations cannot be used at the same time."

        if prompt is None:
            # apply sft format
            sft_format = self.apply_sft_template_for_multi_turn_prompts(
                conversations=conversations,
                sft_format=self.sft_format,
                system_prompt=self.system_prompt,
            )
        else:
            sft_format = prompt
        # sft_format: 'You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.\n\nUser: <image_placeholder>Describe this image.\n\nAssistant: Stop<｜end▁of▁sentence｜>'
        conv_template = get_conv_template(self.sft_format)
        roles = conv_template.roles
        # tokenize todo debug!
        # input_ids: torch.tensor([xxxx]), num_image_tokens -> 576
        input_ids, num_image_tokens = self._tokenizer_and_add_image_token(sft_format)

        sft_format = [sft_format]
        input_ids = input_ids.unsqueeze(dim=0)  # [1, len]
        targets = input_ids.clone()
        sep = roles[1] + ": "  # Assistant:
        for conversation, target in zip(sft_format, targets):
            rounds = conversation.split(conv_template.sep2)  # 划分每轮对话
            cur_len = 1
            target[:cur_len] = self.ignore_id
            for i, rd in enumerate(rounds):
                parts = rd.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep

                round_len = len(self._tokenizer_and_add_image_token(rd)[0])
                instruction_len = len(self._tokenizer_and_add_image_token(parts[0])[0])

                target[cur_len : cur_len + instruction_len - 2] = self.ignore_id

                cur_len += round_len
            target[cur_len:] = self.ignore_id

        # load images
        images_outputs = self.image_processor(images, return_tensors="pt")

        if not self.is_train:
            prepare = VLChatProcessorOutput(
                sft_format=sft_format[0],
                input_ids=input_ids[0],
                pixel_values=images_outputs.pixel_values[0],
                num_image_tokens=num_image_tokens,
            )
        else:
            prepare = VLChatProcessorTrainOutput(
                sft_format=sft_format[0],
                input_ids=input_ids[0],
                labels=targets[0],
                pixel_values=images_outputs.pixel_values[0],
                num_image_tokens=num_image_tokens,
            )

        return prepare

    def __call__(
        self,
        *,
        prompt: str = None,
        conversations: list[dict[str, str]] = None,
        images: list[Image] = None,
        force_batchify: bool = True,
        **kwargs,
    ):
        """

        Args:
            prompt (str): the formatted prompt;
            conversations (List[Dict]): conversations with a list of messages;
            images (List[ImageType]): the list of images;
            force_batchify (bool): force batchify the inputs;
            **kwargs:

        Returns:
            outputs (BaseProcessorOutput): the output of the processor,
                - input_ids (torch.LongTensor): [N + image tokens]
                - images (torch.FloatTensor): [n_images, 3, H, W]
                - image_id (int): the id of the image token
                - num_image_tokens (List[int]): the number of image tokens
        """
        prepare = self.process_one(
            prompt=prompt, conversations=conversations, images=images
        )

        if force_batchify:
            prepare = self.batchify([prepare])

        return prepare

    def batchify(
        self, prepare_list: list[VLChatProcessorOutput]
    ) -> BatchedVLChatProcessorOutput | BatchedVLChatProcessorTrainOutput:
        """
        Preprocesses the inputs for multimodal inference.

        Args:
            prepare_list (List[VLChatProcessorOutput]): A list of VLChatProcessorOutput.

        Returns:
            BatchedVLChatProcessorOutput: A dictionary of the inputs to use for multimodal inference.
        """

        batch_size = len(prepare_list)
        sft_format = []
        n_images = []
        seq_lens = []
        for prepare in prepare_list:
            n_images.append(len(prepare.num_image_tokens))
            seq_lens.append(len(prepare))

        input_token_max_len = max(seq_lens)
        max_n_images = max(1, max(n_images))

        batched_input_ids = torch.full(
            (batch_size, input_token_max_len), self.pad_id
        ).long()  # FIXME

        batched_labels = torch.full(
            (batch_size, input_token_max_len), self.ignore_id
        ).long()

        batched_attention_mask = torch.zeros((batch_size, input_token_max_len)).long()
        batched_pixel_values = torch.zeros(
            (batch_size, max_n_images, *self.image_processor.default_shape)
        ).float()
        batched_images_seq_mask = torch.zeros((batch_size, input_token_max_len)).bool()
        batched_images_emb_mask = torch.zeros(
            (batch_size, max_n_images, self.num_image_tokens)
        ).bool()

        batched_history = None
        if prepare_list[0].history is not None:
            history_actions = [p["history"]["actions"] for p in prepare_list]
            history_action_lens = [288]  # make max_len to 288 -> 288*2 = 576
            for ha in history_actions:
                history_action_lens.extend([len(h) for h in ha])
            history_actions_input_ids = torch.full(
                (batch_size, len(history_actions[0]), max(history_action_lens)),
                self.pad_id,
            ).long()
            for bs in range(batch_size):
                for i, ha in enumerate(history_actions[bs]):
                    history_actions_input_ids[bs, i, -len(ha) :] = torch.LongTensor(ha)
            batched_history = {
                "images": torch.cat(
                    [p["history"]["images"].unsqueeze(0) for p in prepare_list], dim=0
                ),
                "actions": history_actions_input_ids,
            }

        for i, prepare in enumerate(prepare_list):
            input_ids = prepare.input_ids
            labels = prepare.labels
            seq_len = len(prepare)
            n_image = len(prepare.num_image_tokens)
            # left-padding
            batched_attention_mask[i, -seq_len:] = 1
            batched_input_ids[i, -seq_len:] = torch.LongTensor(input_ids)
            batched_labels[i, -seq_len:] = torch.LongTensor(labels)
            batched_images_seq_mask[i, -seq_len:] = input_ids == self.image_id

            if n_image > 0:
                batched_pixel_values[i, :n_image] = prepare.pixel_values
                for j, n_image_tokens in enumerate(prepare.num_image_tokens):
                    batched_images_emb_mask[i, j, :n_image_tokens] = True

            sft_format.append(prepare.sft_format)

        if not self.is_train:
            batched_prepares = BatchedVLChatProcessorOutput(
                input_ids=batched_input_ids,
                attention_mask=batched_attention_mask,
                pixel_values=batched_pixel_values,
                images_seq_mask=batched_images_seq_mask,
                images_emb_mask=batched_images_emb_mask,
                sft_format=sft_format,
                history=batched_history,
            )
        else:
            batched_prepares = BatchedVLChatProcessorTrainOutput(
                input_ids=batched_input_ids,
                labels=batched_labels,
                attention_mask=batched_attention_mask,
                pixel_values=batched_pixel_values,
                images_seq_mask=batched_images_seq_mask,
                images_emb_mask=batched_images_emb_mask,
                sft_format=sft_format,
                history=batched_history,
            )
        return batched_prepares

    def _tokenizer_and_add_image_token(self, prompt: str):
        input_ids = self.tokenizer.encode(prompt)
        input_ids = torch.LongTensor(input_ids)

        # add image tokens to the input_ids
        image_token_mask: torch.BoolTensor = input_ids == self.image_id
        image_indices = image_token_mask.nonzero()  # 图像所在的token index
        input_ids, num_image_tokens = self.add_image_token(
            image_indices=image_indices,
            input_ids=input_ids,
        )
        return input_ids, num_image_tokens
