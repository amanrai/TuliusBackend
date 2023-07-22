"""
Dataset Preprocessors
"""

from config import *

def processDataOpenAssistant(ds):
    print("Processing OpenAssistant")
    messages = {}
    for row in ds:
        messages[row["message_id"]] = {}
        messages[row["message_id"]]["message"] = row
        messages[row["message_id"]]["children"] = []

    if (row["role"] != "assistant"):
        print(row)
        
    for row in ds:
        if row["parent_id"] != None:
            messages[row["parent_id"]]["children"].append(row)
        
    
    conversations = {}
    print(len(messages))
        

def processPreMadeOpenAssistantDataset(ds):
    texts = []
    for row in ds:
        _t = row["text"].replace("User: User:", "User:").replace("Assistant: Chip:", "Assistant:")
        _t = _t.replace("Assistant:", "\nAssistant:")
        _t += stop_token
        texts.append(_t)
    return texts
        
def processh2oOpenAssistant(ds):
    texts = []
    for row in ds:
        _t = row["input"].replace("<human>:", "User:").replace("<bot>:", "Assistant:")
        _t += stop_token
        texts.append(_t)
        # print(row["text"])
    return texts

def processDatabricks(ds):
    texts = []
    for row in ds:
        include_context = False
        if (len(row["context"]) > 0):
            include_context = True
        instruction = row["instruction"]
        context = row["context"]
        response = row["response"]
        if (include_context):
            _t = f"User: {instruction}\nSystem: Here is some additional information:\n{context}\nAssistant: {response}"
        else:
            _t = f"User: {instruction} \nAssistant: {response}"
        _t += stop_token
        texts.append(_t)
        # print(_t)
    return texts

def processAlpaca(ds):
    texts = []
    for row in ds:
        include_context = False
        if (len(row["input"]) > 0):
            include_context = True
        instruction = row["instruction"]
        context = row["input"]
        response = row["output"]
        if (include_context):
            _t = f"User: {instruction}\n{context}\nAssistant: {response}"
        else:
            _t = f"User: {instruction} \nAssistant: {response}"
        _t += stop_token
        texts.append(_t)
        # print(_t)
    return texts

def processAnthropic(ds):
    texts = []
    for row in ds:
        _t = row["chosen"].replace("Human:", "User:")
        _t += stop_token
        # print(_t)
        texts.append(_t)
    return texts

def processInstructGPTJ(ds):
    texts = []
    for row in ds:
        instruction = row["prompt"]
        output = row["chosen"]
        _t = f"User: {instruction}\nAssistant: {output}"
        _t += stop_token
        # print(_t)
        texts.append(_t)
    return texts

def processEssayInstructions(ds):
    texts = []
    for row in ds:
        instruction = row["prompt"].replace("Human:", "User:")
        output = row["chosen"]
        _t = f"{instruction}\nAssistant:{output}"
        _t += stop_token
        # print(_t)
        texts.append(_t)
    return texts

def processELI5(ds):
    texts = []
    for row in ds:
        if (not row["human_answers"].lower().startswith("edit")):
            instruction = row["question"]
            output = row["human_answers"]
            _t = f"User: {instruction}\nAssistant: {output}"
            _t += stop_token
            texts.append(_t)
            print(_t)
    return texts

def processAlpacaCode(ds):
    texts = []
    for row in ds:
        instruction = row["prompt"]
        output = row["completion"]
        _t = f"User: {instruction}\nAssistant: {output}"
        _t += stop_token
        texts.append(_t)

    return texts