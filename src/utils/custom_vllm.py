"""
This is a customized version of the vLLM LLM class.

Additional features:
- ...: allow the user to mannually insert a meta reasoning template (e.g. a coping strategy) as a part of the CoT reasoning process. This would allow the model to continue reasoning with the predefined template. 
"""

import yaml
from jinja2 import Template
from typing import Any, Optional, Union, cast

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

class CustomLLM(LLM):

    def __init__(self, *args, **kwargs):

        coping_chat_template_path = kwargs.get(
            'coping_chat_template_path', 
            './prompts/coping_strategies.yaml'
        )
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

        # remove custom kwargs
        kwargs = {
            k: v for k, v in kwargs.items() if k != 'coping_chat_template_path'
        } 

        super().__init__(*args, **kwargs)

    
    def _make_coping_chat_messages(
        self, 
        situation_desc_list: list,
        patient_thought_list: list,
        patient_persona_profile_list: list,
    ) -> list[dict[str, list[ChatCompletionMessageParam]]]:

        coping_chat_messages = []
        for situation, thought, persona_profile in zip(
            situation_desc_list,
            patient_thought_list,
            patient_persona_profile_list,
        ):
            # first, organize the peronsa profile dictionary to something more readable
            persona_profile_desc = (
                persona_profile['persona_hub'] + '\n\nDetailed Persona Profile:\n'
                f'Name: {persona_profile["name"]}\n'
                f'Gender: {persona_profile["gender"]}\n'
                f'Occupation: {persona_profile["occupation"]}\n'
                f'Education: {persona_profile["education"]}\n'
                f'Personality: {persona_profile["traits"]}\n'
            )

            generic_instruction_prompt = self.coping_generic_instruction_template.render(
                situation=situation,
                thought=thought,
                persona_profile=persona_profile_desc.strip(),
            )

            generic_thought_prompt = self.coping_generic_thought_template.render(
                situation=situation,
                thought=thought,
                persona_profile=persona_profile_desc.strip(),
            )

            user_specific_coping_msg_dict = {}
            for strategy_name, strategy_template in self.coping_strategy_template.items():
                user_specific_coping_msg_dict[strategy_name] = [
                    {'role': 'system', 'content': self.coping_system_prompt},
                    {'role': 'user', 'content': generic_instruction_prompt},
                    {'role': 'assistant', 'content': generic_thought_prompt + '\n' + strategy_template},
                ]

            coping_chat_messages.append(user_specific_coping_msg_dict)

        return coping_chat_messages


    def coping_chat(
        self,
        situation_desc_list: list,
        patient_thought_list: list,
        patient_persona_profile_list: list,
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
            patient_persona_profile_list=patient_persona_profile_list,
        )

        # flatten the coping chat messages while keep track of the sample index and key
        messages = []
        sample_idx_key_list = []
        for sample_idx, coping_chat_msg_dict in enumerate(coping_chat_messages):
            for coping_strategy_name, coping_strategy_msg_list in coping_chat_msg_dict.items():
                messages.append(coping_strategy_msg_list)
                sample_idx_key_list.append((sample_idx, coping_strategy_name))

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
                prompt_instruction, coping_strategy_content = prompt_str.split('<think>')

                coping_strategy_content = coping_strategy_content.split('</think>')[-1].split('<|im_end|>')[0].strip()
                coping_strategy_content += '\n\n' + self.coping_postfix
                prompt_str = prompt_instruction.strip() + '\n<think>\n' + coping_strategy_content + '\n</think>'

                print(prompt_str)
                raise SystemExit

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

        return self.generate(
            prompts,
            sampling_params=sampling_params,
            use_tqdm=use_tqdm,
            lora_request=lora_request,
        )