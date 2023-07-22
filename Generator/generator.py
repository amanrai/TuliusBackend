#Aman Rai, July 2023
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList
from optimum.bettertransformer import BetterTransformer

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops = []):
      super().__init__()
      self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, stops = []):
        for stop in self.stops:
            _f = input_ids[0][-len(stop):]            
            _o = _f == stop
            if torch.all(_o):
                return True

        return False

class Generator:

    def __init__(self, model, tokenizer, half_mode=torch.float16):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        with open("./stop_words.txt", "r") as f:
            self.stopping_words = f.read().split("\n")
        stop_tokens = [torch.LongTensor(self.tokenizer.encode(w)).to(self.device) for w in self.stopping_words]
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops = stop_tokens)])        
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=half_mode, trust_remote_code=True)
        self.model = BetterTransformer.transform(self.model)
        self.model = self.model.to(self.device)
        self.model.eval()        
        
    def generate(self, prompt, decode_args = {}):
        if ("max_new_tokens" not in decode_args):
            decode_args["max_new_tokens"] = 128
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        input_length = inputs.input_ids.shape[1]
        outputs = self.model.generate(
            **inputs, 
            **decode_args,
            pad_token_id=self.tokenizer.eos_token_id,
            stopping_criteria=self.stopping_criteria,
            return_dict_in_generate=True
        )
        token = outputs.sequences[0, input_length:]
        output_str = self.tokenizer.decode(token)
        for stop in self.stopping_words:
            output_str = output_str.replace(stop, "")
        return output_str
        

