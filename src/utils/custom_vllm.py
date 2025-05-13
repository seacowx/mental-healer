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
        self.coping_chat_template_dict = {
            k: Template(v) for k, v in self.coping_chat_template_dict.items()
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
    ) -> list[list[ChatCompletionMessageParam]]:

        generic_thought = self.coping_chat_template_dict['generic_thought']

        for situation, thought, persona_profile in zip(
            situation_desc_list,
            patient_thought_list,
            patient_persona_profile_list,
        ):
            generic_thought_prompt = generic_thought.render(
                situation=situation,
                thought=thought,
                persona_profile=persona_profile,
            )

            print(generic_thought_prompt)
            raise SystemExit

        raise NotImplementedError()


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
        messages = self._make_coping_chat_messages(
            situation_desc_list=situation_desc_list,
            patient_thought_list=patient_thought_list,
            patient_persona_profile_list=patient_persona_profile_list,
        )

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
            model_config,
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
                    model_config,
                    tokenizer,
                    conversation=conversation,
                    **_chat_template_kwargs,
                )
                # Special tokens are already included in chat templates so
                # should not be added by the tokenizer in this case.
                prompt_token_ids = tokenizer.encode(prompt_str,
                                                    add_special_tokens=False)

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