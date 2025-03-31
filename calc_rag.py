from transformers import LogitsProcessor
import torch
import re

class InferenceRAGLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, retriever):
        self.tokenizer = tokenizer
        self.calc_function = retriever

        self.active_calc = None
        self.inject_result = False
        self.result_tokens = None
        self.inject_index = 0
        self.calc_cache = {}     

    def __call__(self, input_ids, scores):
        prompt_ids = tokenizer(input_text, return_tensors="pt")["input_ids"][0]
        prompt_len = prompt_ids.shape[0]
        generated_ids = input_ids[0][prompt_len:] 
        text_so_far = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        # Extract the latest calculation query
        calc_matches = re.findall(r"\[CALC:\s*([\d+\-*/(). ]+?)\s*:\]", text_so_far)
        result_matches = re.findall(r"\[RESULT:\s*([0-9.eE+-]+)\s*:\]", text_so_far)
        if calc_matches and len(calc_matches)!=len(result_matches):
            last_calc = calc_matches[-1].strip()
            if last_calc != self.active_calc:
                self.active_calc = last_calc
                if last_calc in self.calc_cache:
                    result = self.calc_cache[last_calc]
                else:
                    try:
                        result = self.calc_function(last_calc)
                        #777275194773.0
                    except Exception:
                        result = "ERROR"
                    if len(result) > 15:
                        result = "OVERFLOW"
                    # Save in cache
                    self.calc_cache[last_calc] = result
                result_str = f" [RESULT:{result}:]"
                self.result_tokens = self.tokenizer.encode(result_str, add_special_tokens=False)
                self.inject_index = 0
                self.inject_result = True

        # Ensure proper result injection
        if  self.inject_result and self.result_tokens and self.inject_index < len(self.result_tokens):
            scores[:, self.result_tokens[self.inject_index ]] += 100  # Bias toward `[RESULT:` token
            self.inject_index += 1
            if self.result_tokens and self.inject_index >= len(self.result_tokens):
                calc_matches = None
                self.result_tokens = []
                self.inject_index = 0
                self.active_calc = None  # 允许处理下一个 CALC 表达式
                self.inject_result = False
            return scores
        
        return scores
    

from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList


import sympy


def calc_sympy(query):
    
    r = str(sympy.sympify(query))
    return r


modelname = "google/gemma-2-2b-it"

tokenizer = AutoTokenizer.from_pretrained(modelname)
model = AutoModelForCausalLM.from_pretrained(
    modelname,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)


logits_processor = InferenceRAGLogitsProcessor(tokenizer, calc_sympy)
logits_processor_list = LogitsProcessorList([logits_processor])


input_text = """
You are a helpful calculation assistant to do multi-step calculations.

!Important: for calculation, use the calculation tool by using ``` [CALC:information to query:] ``` in your text to get accurate calculation result, which will be supplied in ``` [RESULT:calculation result:] ``` .
Notice the space before and after the square brackets, do not use new line, also do not use surrounding '```'. **Do not use '[CALC::]' and '[RESULT::]' for your reasoning if it is not about using calculation tool.**

For example:
Example Message: calculate 3593 * 476787 and calculate the third rightmost digit of the result multiplied by 564457.
Example Response: [CALC:3593 * 476787:] [RESULT:1713095691:] , the third rightmost digit of the result is 6, [CALC:6 * 564457:] [RESULT:3386742:] , therefore the final result is 3386742.

---

Message: calculate 786989 * 987657 and calculate the leftmost digit of the result multiplied by 12349.

Provide your response below:
"""


input_ids = tokenizer(input_text, return_tensors="pt")

from transformers import TextStreamer
streamer = TextStreamer(tokenizer, skip_prompt=True)

outputs = model.generate(**input_ids, logits_processor=logits_processor_list, max_new_tokens=4000, streamer=streamer)
tokenizer.decode(outputs[0])
