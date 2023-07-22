#Aman Rai, July 2023
from config import *
from datasets import load_dataset, concatenate_datasets
from datasets import DatasetDict, Dataset

datasets = {
    "birgermoell/open_assistant_dataset":processPreMadeOpenAssistantDataset,
    "h2oai/openassistant_oasst1":processh2oOpenAssistant, #this may overlap some of the stuff above.
    "iamketan25/essay-instructions-dataset":processEssayInstructions,
    "Anthropic/hh-rlhf":processAnthropic,
    "hakurei/open-instruct-v1":processAlpaca, #this contains a lot of other datasets.
    "HuggingFaceH4/CodeAlpaca_20K":processAlpacaCode,
    "quac":processQUAC,
    "local/SQuAD_prepared.json": processSQuAD,
}

all_texts = []

for dataset in datasets:
    print(dataset)
    _fn = datasets[dataset]

# print(len(all_texts))

rows = [{"text":_t} for _t in all_texts]
f_dataset_json = {
    "version":"0.1.0",
    "data":rows
}

with open("./instruct_dataset_compiled.json", "w") as f:
    f.write(json.dumps(f_dataset_json, indent=4))