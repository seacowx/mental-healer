"""
This is a customized version of the vLLM LLM class.

Additional features:
- ...: allow the user to mannually insert a meta reasoning template (e.g. a coping strategy) as a part of the CoT reasoning process. This would allow the model to continue reasoning with the predefined template. 
"""

import yaml
import warnings
from jinja2 import Template
from typing import Any, Optional, Union, cast

import torch
from vllm import LLM
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam,
    ChatTemplateContentFormatOption,
    resolve_chat_template_content_format,
    parse_chat_messages,
    apply_hf_chat_template,
    apply_mistral_chat_template,
)
from vllm.transformers_utils.tokenizer import MistralTokenizer
from vllm.utils import is_list_of
from vllm.inputs import TokensPrompt, TextPrompt

from utils.persona_utils import verbalize_persona_profile
from utils.therapeutic_utils import TherapeuticSessionBuffer


class CustomLLM(LLM):

    def __init__(self, *args, **kwargs):

        coping_chat_template_path = kwargs.get(
            'coping_chat_template_path', 
            ''
        )
        if coping_chat_template_path:
            self.coping_chat_template_dict = yaml.safe_load(
                open(coping_chat_template_path)
            )
            self.coping_generic_instruction_template = Template(self.coping_chat_template_dict['generic_instruction'])
            self.coping_generic_thought_template = Template(self.coping_chat_template_dict['generic_thought'])
            self.coping_system_prompt = self.coping_chat_template_dict['system']
            self.coping_postfix = self.coping_chat_template_dict['coping_postfix']
            self.coping_strategy_template = {
                k: v for k, v in self.coping_chat_template_dict.items() 
                if k not in  ['generic_thought', 'generic_instruction', 'coping_postfix', 'system']
            }
        else:
            warnings.warn('No coping chat template path provided. Coping Chat cannot be used.')

        # remove custom kwargs
        kwargs = {
            k: v for k, v in kwargs.items() if k != 'coping_chat_template_path'
        } 

        # wrap device to torch.device
        if kwargs.get('device'):
            kwargs['device'] = torch.device(kwargs['model_device'])

        super().__init__(*args, **kwargs)

    
    def _make_coping_chat_messages(
        self, 
        situation_desc_list: list,
        patient_thought_list: list,
        patient_persona_profile_desc_list: list,
        active_sample_idx_list: list[int],
        active_coping_strategy_idx_list: list[list[int]],
    ) -> list[dict[str, list[ChatCompletionMessageParam]]]:
        """
        Make coping chat messages for each sample in the batch.

        Iterate throught each sample and each coping strategy. Make coping caht message according to the previous patient's thought.
        Coping strategies are sampled from the 2x2x2 grid from the reAppraisal framework

        Returns:
            coping_chat_messages: list[dict[str, list[ChatCompletionMessageParam]]]: batched messages for each sample in the batch. Inactive coping strategies are indicated by an empty list.
        """

        coping_chat_messages = []
        for sample_idx, (situation, thought_list, persona_profile_desc) in enumerate(zip(
            situation_desc_list,
            patient_thought_list,
            patient_persona_profile_desc_list,
        )):

            if sample_idx not in active_sample_idx_list:
                coping_chat_messages.append({})
                continue

            user_specific_coping_msg_dict = {}
            for strategy_idx, (strategy_name, strategy_template) in enumerate(self.coping_strategy_template.items()):

                cur_thought = thought_list[strategy_idx]

                generic_instruction_prompt = self.coping_generic_instruction_template.render(
                    situation=situation,
                    thought=cur_thought,
                    persona_profile=persona_profile_desc.strip(),
                )
                generic_thought_prompt = self.coping_generic_thought_template.render(
                    situation=situation,
                    thought=cur_thought,
                    persona_profile=persona_profile_desc.strip(),
                )

                if strategy_idx not in active_coping_strategy_idx_list[sample_idx]:
                    user_specific_coping_msg_dict[strategy_name] = []
                else:
                    user_specific_coping_msg_dict[strategy_name] = [
                        {'role': 'system', 'content': self.coping_system_prompt},
                        {'role': 'user', 'content': generic_instruction_prompt},
                        {'role': 'assistant', 'content': generic_thought_prompt + '\n\n' + strategy_template},
                    ]

            coping_chat_messages.append(user_specific_coping_msg_dict)

        return coping_chat_messages


    def coping_chat(
        self,
        situation_desc_list: list,
        patient_thought_list: list,
        patient_persona_profile_desc_list: list,
        session_buffer: TherapeuticSessionBuffer,
        active_sample_idx_list: list[int],
        active_coping_strategy_idx_list: list[list[int]],
        sampling_params: Optional[Union[SamplingParams,
                                        list[SamplingParams]]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[LoRARequest] = None,
        chat_template: Optional[str] = None,
        chat_template_content_format: ChatTemplateContentFormatOption = "auto",
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        tools: Optional[list[dict[str, Any]]] = None,
        chat_template_kwargs: Optional[dict[str, Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
    ):

        # make coping chat messages
        coping_chat_messages = self._make_coping_chat_messages(
            situation_desc_list=situation_desc_list,
            patient_thought_list=patient_thought_list,
            patient_persona_profile_desc_list=patient_persona_profile_desc_list,
            active_sample_idx_list=active_sample_idx_list,
            active_coping_strategy_idx_list=active_coping_strategy_idx_list,
        )

        # flatten the coping chat messages while keep track of the sample index and key
        messages = []
        sample_idx_key_list = []
        for sample_idx, coping_chat_msg_dict in enumerate(coping_chat_messages):

            if not coping_chat_msg_dict:
                continue

            for coping_strategy_name, coping_strategy_msg_list in coping_chat_msg_dict.items():
                if coping_strategy_msg_list:
                    messages.append(coping_strategy_msg_list)
                    sample_idx_key_list.append((str(sample_idx), coping_strategy_name))

        list_of_messages: list[list[ChatCompletionMessageParam]]

        # Handle multi and single conversations
        if is_list_of(messages, list):
            # messages is list[list[...]]
            list_of_messages = cast(list[list[ChatCompletionMessageParam]],
                                    messages)
        else:
            # messages is list[...]
            list_of_messages = [
                cast(list[ChatCompletionMessageParam], messages)
            ]

        tokenizer = self.get_tokenizer(lora_request)

        model_config = self.llm_engine.get_model_config()
        resolved_content_format = resolve_chat_template_content_format(
            chat_template,
            tools,
            chat_template_content_format,
            tokenizer,
        )

        _chat_template_kwargs: dict[str, Any] = dict(
            chat_template=chat_template,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tools=tools,
        )
        _chat_template_kwargs.update(chat_template_kwargs or {})

        prompts: list[Union[TokensPrompt, TextPrompt]] = []

        for msgs in list_of_messages:
            # NOTE: _parse_chat_message_content_parts() currently doesn't
            # handle mm_processor_kwargs, since there is no implementation in
            # the chat message parsing for it.

            conversation, mm_data = parse_chat_messages(
                msgs,
                model_config,
                tokenizer,
                content_format=resolved_content_format,
            )

            if isinstance(tokenizer, MistralTokenizer):
                prompt_token_ids = apply_mistral_chat_template(
                    tokenizer,
                    messages=msgs,
                    **_chat_template_kwargs,
                )
            else:
                prompt_str = apply_hf_chat_template(
                    conversation=conversation,
                    tokenizer=tokenizer,
                    **_chat_template_kwargs,
                )

                # modify the prompt to put the coping strategy content in between the <think> and </think> tags
                prompt_instruction, coping_strategy_content = prompt_str.rsplit('<think>', 1)

                coping_strategy_content = coping_strategy_content \
                    .rsplit('</think>', 1)[-1] \
                    .rsplit('<|im_end|>', 1)[0].strip()
                coping_strategy_content += '\n\n' + self.coping_postfix
                prompt_str = prompt_instruction.strip() + '\n<think>\n' + coping_strategy_content + '\n</think>'

                # Special tokens are already included in chat templates so
                # should not be added by the tokenizer in this case.
                prompt_token_ids = tokenizer.encode(
                    prompt_str,
                    add_special_tokens=False
                )

            prompt = TokensPrompt(prompt_token_ids=prompt_token_ids)

            if mm_data is not None:
                prompt["multi_modal_data"] = mm_data

            if mm_processor_kwargs is not None:
                prompt["mm_processor_kwargs"] = mm_processor_kwargs

            prompts.append(prompt)

        output_list = self.generate(
            prompts,
            sampling_params=sampling_params,
            use_tqdm=use_tqdm,
            lora_request=lora_request,
        )

        return output_list, sample_idx_key_list