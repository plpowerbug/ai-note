from transformers import LogitsProcessor
import torch
import re

class InferenceRAGLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, retriever, boost=30.0):
        self.tokenizer = tokenizer
        self.calc_function = retriever
        self.boost = boost
        self.calc_to_result_token_ids = {}

    def set_prompt(self, prompt):
        # 提取所有 [CALC: ... :] 并计算结果
        pattern = r"\[CALC:(.+?):\]"
        matches = re.findall(pattern, prompt)
        for query in matches:
            query_clean = query.strip()
            try:
                result = self.calc_function(query_clean)
                result_str = f"[RESULT:{result}:]"
                result_token_ids = self.tokenizer.encode(result_str, add_special_tokens=False)
                self.calc_to_result_token_ids[query_clean] = result_token_ids
            except Exception as e:
                print(f"⚠️ Error calculating '{query_clean}': {e}")

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 获取当前生成的文本
        text_so_far = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

        # 找到最近的 [CALC: ... :]
        calc_matches = re.findall(r"\[CALC:(.+?):\]", text_so_far)
        if not calc_matches:
            return scores

        latest_query = calc_matches[-1].strip()
        if latest_query not in self.calc_to_result_token_ids:
            return scores

        expected_ids = self.calc_to_result_token_ids[latest_query]
        generated_ids = input_ids[0].tolist()

        # 对比当前生成位置，尝试引导模型生成 result token
        result_so_far = []
        for i in reversed(range(len(generated_ids))):
            result_so_far.insert(0, generated_ids[i])
            if result_so_far == expected_ids[:len(result_so_far)]:
                break

        next_index = len(result_so_far)
        if next_index < len(expected_ids):
            next_token_id = expected_ids[next_index]
            scores[0, next_token_id] += self.boost

        return scores


# ======================
# 🧮 sympy 计算函数
# ======================
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


logits_processor.set_prompt(input_text)
logits_processor_list = LogitsProcessorList([logits_processor])
input_ids = tokenizer(input_text, return_tensors="pt")

from transformers import TextStreamer
streamer = TextStreamer(tokenizer, skip_prompt=True)

outputs = model.generate(**input_ids, logits_processor=logits_processor_list, max_new_tokens=4000, streamer=streamer)
tokenizer.decode(outputs[0])
