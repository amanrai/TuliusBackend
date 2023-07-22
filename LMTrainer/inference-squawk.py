#Aman Rai, July 2023.
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

model = AutoModelForCausalLM.from_pretrained("./sQuAwk-opt-125m-sQuAwk/checkpoint-7800")
tokenizer = AutoTokenizer.from_pretrained("./sQuAwk-facebook/opt-125m-customTokenizer")

model.to("cuda")

user_handle = "<|USER|>"
system_handle = "<|SYSTEM|>"
bot_handle = "<|ASSISTANT|>"
instruction = "<|INSTRUCTION|>"
context_token = "<|CONTEXT|>"
answer_not_in_context_token = "<|ANSWER_NOT_IN_CONTEXT|>"
answer_in_context = "<|ANSWER_IN_CONTEXT|>"
eos_token = "</s>"

template =  "{context_token} \n {context} \n {user_handle} {question} \n {bot_handle} "

def extract_from_dp(dp):
    context = dp.split(user_handle)[0].strip().replace(context_token, "").strip()
    question = dp.split(user_handle)[1].split(bot_handle)[0].strip()
    answer = dp.split(bot_handle)[1].strip().replace(eos_token, "").strip()
    return context, question, answer


with open("./qa_basic_dataset.json", "r") as f:
    ds = json.loads(f.read())


for dp in ds["validation"][100:200]:
    c, q, a = extract_from_dp(dp)
    input = template.format(context_token=context_token, context=c, user_handle=user_handle, question=q, bot_handle=bot_handle)
    decode_options = {
        "max_new_tokens": 10,
        "num_beams": 1,
        "temperature": 1.0,
        "num_return_sequences":1
    }
    input_ids = tokenizer.encode(input, return_tensors="pt")
    _out = model.generate(input_ids.to("cuda"), early_stopping=True, **decode_options)
    out = tokenizer.decode(_out[0], skip_special_tokens=True)
    print(out)
    print(f"Actual Answer: {a}")
    print("****")

# print(model)