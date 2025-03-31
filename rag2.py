    def __call__(self, input_ids, scores):
        text_so_far = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        calc_matches = re.findall(r"\[CALC:\s*([\d+\-*/(). ]+?)\s*:\]", text_so_far)
        print(f"Detected CALC queries: {calc_matches}")
        if calc_matches:
            last_calc = calc_matches[-1].strip()
            if last_calc != self.active_calc:
                self.active_calc = last_calc
                try:
                    result = self.calc_function(last_calc)
                except Exception:
                    result = "ERROR"
                if len(result) > 15:
                    result = "OVERFLOW"
                result_str = f" [RESULT:{result}:]"
                self.result_tokens = self.tokenizer.encode(result_str, add_special_tokens=False)
                self.inject_index = 0
        if self.result_tokens and self.inject_index < len(self.result_tokens):
                    next_token_id = self.result_tokens[self.inject_index]
                    # Force the next token in the result sequence
                    forced_scores = torch.full_like(scores, -float('inf'))
                    forced_scores[:, next_token_id] = 0  # Highest score
                    self.inject_index += 1
                    return forced_scores
        return scores



Detected CALC queries: ['3593 * 476787', '6 * 564457']
[RESULT:3386742.0:] Detected CALC queries: ['3593 * 476787', '6 * 564457']
, Detected CALC queries: ['3593 * 476787', '6 * 564457']
the Detected CALC queries: ['3593 * 476787', '6 * 564457']
Detected CALC queries: ['3593 * 476787', '6 * 564457']
leftmost Detected CALC queries: ['3593 * 476787', '6 * 564457']
digit Detected CALC queries: ['3593 * 476787', '6 * 564457']
of Detected CALC queries: ['3593 * 476787', '6 * 564457']
the Detected CALC queries: ['3593 * 476787', '6 * 564457']
result Detected CALC queries: ['3593 * 476787', '6 * 564457']
is Detected CALC queries: ['3593 * 476787', '6 * 564457']
Detected CALC queries: ['3593 * 476787', '6 * 564457']
Detected CALC queries: ['3593 * 476787', '6 * 564457']
3, Detected CALC queries: ['3593 * 476787', '6 * 564457']
Detected CALC queries: ['3593 * 476787', '6 * 564457']
Detected CALC queries: ['3593 * 476787', '6 * 564457']
Detected CALC queries: ['3593 * 476787', '6 * 564457']
[CALC:3 Detected CALC queries: ['3593 * 476787', '6 * 564457']
* Detected CALC queries: ['3593 * 476787', '6 * 564457']
Detected CALC queries: ['3593 * 476787', '6 * 564457']
Detected CALC queries: ['3593 * 476787', '6 * 564457']
Detected CALC queries: ['3593 * 476787', '6 * 564457']
Detected CALC queries: ['3593 * 476787', '6 * 564457']
Detected CALC queries: ['3593 * 476787', '6 * 564457']
Detected CALC queries: ['3593 * 476787', '6 * 564457', '3 * 12349']
12349:] Detected CALC queries: ['3593 * 476787', '6 * 564457', '3 * 12349']
Detected CALC queries: ['3593 * 476787', '6 * 564457', '3 * 12349']
Detected CALC queries: ['3593 * 476787', '6 * 564457', '3 * 12349']
Detected CALC queries: ['3593 * 476787', '6 * 564457', '3 * 12349']
Detected CALC queries: ['3593 * 476787', '6 * 564457', '3 * 12349']
Detected CALC queries: ['3593 * 476787', '6 * 564457', '3 * 12349']
Detected CALC queries: ['3593 * 476787', '6 * 564457', '3 * 12349']
Detected CALC queries: ['3593 * 476787', '6 * 564457', '3 * 12349']
Detected CALC queries: ['3593 * 476787', '6 * 564457', '3 * 12349']
Detected CALC queries: ['3593 * 476787', '6 * 564457', '3 * 12349']
Detected CALC queries: ['3593 * 476787', '6 * 564457', '3 * 12349']
[RESULT:37047.0:] Detected CALC queries: ['3593 * 476787', '6 * 564457', '3 * 12349']
, Detected CALC queries: ['3593 * 476787', '6 * 564457', '3 * 12349']
