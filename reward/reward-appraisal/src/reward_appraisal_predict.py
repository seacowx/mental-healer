import json
from copy import deepcopy
from ....src.utils.llm_inference import LLMInference



class AppraisalPredictor:

    def __init__(self, prompt_template: dict) -> None:
        self.data = json.load(
            open('../../../data/reward-appraisal/envent/envent_organized.json', 'r'),         
        )

        self.prompt_template = prompt_template


    def predict(self):

        appraisal_desc_msg_list = []
        for entry_idx, (id, entry) in enumerate(self.data.items()):
            cur_situation = entry['context'].strip()
            cur_appraisal_prediction_prompt = deepcopy(self.prompt_template)
            cur_appraisal_prediction_prompt[1]['content'] = cur_appraisal_prediction_prompt[1]['content'].replace(
                '{{scenario}}', cur_situation
            )

            appraisal_desc_msg_list.append(cur_appraisal_prediction_prompt)
