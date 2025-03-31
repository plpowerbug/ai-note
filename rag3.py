class InferenceRAGLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, retriever):
        self.tokenizer = tokenizer
        self.calc_function = retriever
        self.calc_start = self.tokenizer.encode(" [CALC:", add_special_tokens=False)
        self.calc_end = self.tokenizer.encode(":]", add_special_tokens=False)
        self.result_start = self.tokenizer.encode(" [RESULT:", add_special_tokens=False)
        self.result_end = self.tokenizer.encode(":]", add_special_tokens=False)
        
        self.active_calc = None
        self.inject_result = False
        self.result_tokens = None

    def __call__(self, input_ids, scores):
        # Decode the generated text so far
        text_so_far = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        print(f"\nGenerated text so far:\n{text_so_far}")

        # Extract the latest calculation query
        calc_matches = re.findall(r"\[CALC:\s*([\d+\-*/(). ]+?)\s*:\]", text_so_far)
        print(f"Detected CALC queries: {calc_matches}")

        if calc_matches:
            last_calc = calc_matches[-1].strip()
            if last_calc != self.active_calc:
                self.active_calc = last_calc
                result = self.calc_function(self.active_calc)

                if len(result) > 15:  # Prevent overflows
                    print(f"⚠️ Overflow detected: {result}, truncating...")
                    result = "OVERFLOW"

                try:
                    self.result_tokens = self.tokenizer.encode(f" [RESULT:{result}:]", add_special_tokens=False)
                    print(f"✅ Calculated result: {result}, Injecting tokens: {self.result_tokens}")
                except Exception as e:
                    print(f"❌ Token encoding failed: {e}")
                    self.result_tokens = None  # Prevent breaking execution

                self.inject_result = True

        # Ensure proper result injection
        if self.inject_result and self.result_tokens:
            scores[:, self.result_tokens[0]] += 100  # Bias toward `[RESULT:` token
            self.inject_result = False  # Reset after injection

        return scores


Provide your response below:
 [CALC:786989 * 987657:] [RESULT:7749999999999999
Detected CALC queries: ['3593 * 476787', '6 * 564457', '786989 * 987657']



