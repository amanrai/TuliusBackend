# Aman Rai, July 2023
# Useful links
# https://github.com/nichtdax/awesome-totally-open-chatgpt
# https://github.com/nomic-ai/gpt4all/tree/main/gpt4all-training
# https://github.com/yaodongC/awesome-instruction-dataset

from config import *
from datasets import load_dataset, concatenate_datasets
from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTNeoXForCausalLM
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding, DataCollatorForLanguageModeling
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftConfig
from torch import nn
import json
from tqdm import trange
import os
import torch
import time
import os
import distutils
from argparse import ArgumentParser

def str2bool(v):
    if (type(v) == bool):
        return v
    return bool(distutils.util.strtobool(v))

parser = ArgumentParser()
parser.add_argument("--lora", type=bool, default=False)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--all_devices_visible", type=bool, default=True)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
parser.add_argument("--baseModel", type=str, default=base_model)
parser.add_argument("--forceRecreateDataset", type=bool, default=forceRebuild)
parser.add_argument("--cpuMaxMemoryGB", type=str, default="32GB")
parser.add_argument("--gpuMaxMemoryGB", type=str, default="16GB")
parser.add_argument("--max_sequence_length", type=int, default=max_sequence_length)
parser.add_argument("--multiGPUIfAvailable", type=bool, default=multiGPUIfAvailable)
parser.add_argument("--runName", type=str, default="sQuAwk")
parser.add_argument("--enableFSDP", type=bool, default=True)
parser.add_argument("--tf32", type=str, default=True)
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--datasetOnly", type=str, default="False")

#Provides the number of GPUs
devices = torch.cuda.device_count()
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(base_model)

_add = list(special_tokens.values())
_add.append(eos_token)
tokenizer.add_tokens(_add)

for token in special_tokens.values():
    print(token, tokenizer.encode(token))

print(eos_token, tokenizer.encode(eos_token))
print(tokenizer.eos_token)

tokenizer.save_pretrained(f"./{args.runName}-{base_model}-customTokenizer")

if (not args.all_devices_visible):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if (str2bool(args.tf32)):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

stop_token = eos_token
tokenizer.pad_token = tokenizer.eos_token

ds_list = []
forceRebuild = args.forceRecreateDataset
max_sequence_length=args.max_sequence_length
doLora = args.lora
base_model = args.baseModel

with open("qa_basic_dataset.json") as f:
    data = json.load(f)

splits = ["train", "validation"]

data["train"] = [{"text": item.replace("<|endoftext|>", tokenizer.eos_token)} for item in data["train"]]
data["validation"] = [{"text": item.replace("<|endoftext|>", tokenizer.eos_token)} for item in data["validation"]]

with open("hf_dataset.json", "w") as f:
    f.write(json.dumps(data))

dataset = load_dataset("json", data_files="hf_dataset.json", field="train")
val = load_dataset("json", data_files="hf_dataset.json", field="validation")

print(dataset)
print(val)

#Dataset Tokenization
def tokenize_map(row):
    return tokenizer(row["text"], max_length=max_sequence_length, truncation=True)

tokenized_validation = val.map(tokenize_map, batched=True)
tokenized_train = dataset.map(tokenize_map, batched=True)

import random
for i in range(10):
    print(random.choices(tokenized_validation["train"]))
    print("****")

# #Dataset Tokenization
# print(f"Tokenizer Loaded: {base_model}")

# tokenized = dataset.map(tokenize_map, batched=True)
# print(tokenized)

#Model Loading
model = AutoModelForCausalLM.from_pretrained(base_model)
print(model)
print("Model Loaded.")
# model.to("cuda")

# if (doLora):
#     for param in model.parameters():
#         param.requires_grad = False  # freeze the model - train adapters later
#         if param.ndim == 1:
#             # cast the small parameters (e.g. layernorm) to fp32 for stability
#             param.data = param.data.to(torch.float32)

#     model.gradient_checkpointing_enable()  # reduce number of stored activations
#     model.enable_input_require_grads()
#     model.config.use_cache = False

#     print("Model prepared for LoRA")

#     #lora Config
#     max_memory = {0: args.gpuMaxMemoryGB, "cpu": args.cpuMaxMemoryGB}
#     peft_model_id = base_model + "_LoRA"
#     config = LoraConfig( r=16, #attention heads
#         lora_alpha=32, #alpha scaling
#         lora_dropout=0.05, #dropouts
#         bias="none",
#         task_type="CAUSAL_LM" # set this for CAUSAL LANGUAGE MODELS (like Bloom, LLaMA) or SEQ TO SEQ (like FLAN, T5)
#     )

#     model = get_peft_model(model, config)
#     print("LoRA model init'd")

# def get_trainable_params(model):
#     """
#     Prints the number of trainable parameters in the model. Borrowed from the net. 
#     """
#     trainable_params = 0
#     all_param = 0
#     for _, param in model.named_parameters():
#         all_param += param.numel()
#         if param.requires_grad:
#             trainable_params += param.numel()
#     print(
#         f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
#     )

# # show trainable params
# get_trainable_params(model)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

print("TF32:", args.tf32)
tf32 = args.tf32 if type(args.tf32 == bool) else distutils.util.strtobool(args.tf32)
_model_name = base_model.split("/")[-1]
_model_name = f"sQuAwk-{_model_name}-{args.runName}"
training_args = TrainingArguments(
    f"{_model_name}",
    evaluation_strategy = "steps",
    learning_rate=args.lr,
    logging_steps = 5,
    weight_decay=0.01,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    eval_steps=150,
    save_steps=150,
    save_total_limit=2,
    num_train_epochs=args.epochs,
    push_to_hub=False,
    load_best_model_at_end=True,
    label_names=["input_ids"],
    # fsdp="shard_grad_op auto_wrap",
    # fsdp_config={"fsdp_transformer_layer_cls_to_wrap" : "GPTNeoXLayer", "fsdp_forward_prefetch": True, "offload_to_cpu": True, "rank0_only": True}
)

# """
# if (devices > 1 and args.enableFSDP):
#     training_args.fsdp = fsdpMode
#     training_args.fsdp_config = fsdpConfig
# elif(devices > 1):
#     training_args.sharedDPP = True
# """

trainer = Trainer(
    model=model, 
    args=training_args,
    tokenizer = tokenizer,
    train_dataset = tokenized_train["train"],
    eval_dataset = tokenized_validation["train"],
    # load_best_model_at_end=True,
    data_collator=data_collator
)

trainer.train()

# model = AutoModelF