base_model = "facebook/opt-125m"
forceRebuild=False
max_sequence_length=1250
multiGPUIfAvailable=True
fsdpIfMultiGPU=False
fsdpConfig= {"fsdp_transformer_layer_cls_to_wrap" : "GPTNeoXLayer"}
fsdpMode = "shard_grad_op auto_wrap"

special_tokens = {
    "user_handle": "<|USER|>",
    "system_handle": "<|SYSTEM|>",
    "bot_handle": "<|ASSISTANT|>",
    "instruction": "<|INSTRUCTION|>",
    "context": "<|CONTEXT|>",
    "answer_not_in_context": "<|ANSWER_NOT_IN_CONTEXT|>",
    "answer_in_context":"<|ANSWER_IN_CONTEXT|>"
}
eos_token = "<|endoftext|>"

possible = "{context_token} \n {context} \n {user_handle} {question} \n {bot_handle} \n {response}{eos_token}"
not_possible = "{context_token} \n {context} \n {user_handle} {question} \n {bot_handle} \n {answer_not_in_context_token} Not answerable. {eos_token}"

def runtimeCustomTemplate(prompt_template, variables):
    if (type(variables) == dict):
        return prompt_template.format(**variables)
    else:
        return ""
