#Aman Rai, July 2023
import torch
import requests
import os
from Prompter import LMPrompter
import json

class LMTask:

    def __init__(self, generator_interface, lm_config_path = "./LMInfo", lm_name=None, lm_config=None):
        assert "uri" in generator_interface
        assert lm_name != None or lm_config != None, "Either the name of the lm or the lm config must be provided"

        self.generator_interface = generator_interface
        generator_info = requests.get(self.generator_interface["uri"]).json()
        self.generator_info = generator_info
        self.generator_info["lm"] = self.generator_info["lm"].split("/")[-1]
        self.generator_info["interface"] = self.generator_interface
        files = os.listdir(lm_config_path)
        print(self.generator_info)
        self.lm_configs = {}
        for file in files:
            if file.endswith(".json"):
                name = file[:-5]
                with open(os.path.join(lm_config_path, file), "r") as f:
                    self.lm_configs[name] = json.loads(f.read())
        
        print(self.lm_configs.keys())
        if (lm_config == None and lm_name not in self.lm_configs):
            assert("Don't have a config for that LM")
        
        if (lm_name != None):
            if (lm_name != self.generator_info["lm"]):
                assert("The backend you provided does not utilize the LM you specified")
        
        if (lm_config == None):
            self.lm_config = self.lm_configs[lm_name]
        else:
            self.lm_config = lm_config
            assert("user_handle" in self.lm_config)
            assert("bot_handle" in self.lm_config)
            assert("system_handle" in self.lm_config)
            assert("instruction_guideline" in self.lm_config)
            assert("decode_configs" in self.lm_config)
            assert("default" in self.lm_config["decode_configs"])
        
        self.prompter_config = self.lm_config["prompter_config"]        
        self.prompter = LMPrompter(**self.prompter_config)
                
    def doTask(self, task_name=None, prompt_variables = None, prompter_config = None, decoder_config=None):
        if (prompter_config is None):
            if (task_name in self.prompter.prompt_choices):
                if (task_name in self.lm_config["decode_configs"]):
                    decode_config = self.lm_config["decode_configs"][task_name]
                else:
                    try:
                        decode_config = self.lm_config["decode_configs"]["default"]
                    except:
                        assert("You must provide a default decode config.")
                prompt = self.prompter.prompt(task_name, **prompt_variables)
            else:
                # decode_config = self.lm_config["decode_configs"]["default"]
                return 404, "Did not find the task you specified, and you did not specify a prompter configuration."            
        else:
            decode_config = self.lm_config["decode_configs"]["default"]
            if ("template" not in prompter_config):
                return 404, "You must provide a custom prompt template"
            if ("variables" not in prompter_config):
                return 404, "You must provide variables for the prompt template"
            prompt = self.prompter.runtimeCustomTemplate(prompter_config["template"], prompter_config["variables"])

        print(prompt)
        post_ = {
            "prompt": prompt,
            "generator_options": decode_config
        }

        response = requests.post(self.generator_interface["uri"] + "/generate", json=post_)
        return response.status_code, response.text

        
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input", type=str, default="What is the capital of Bulgaria?")

    args = parser.parse_args()

    generator_interface = {
        "uri": "http://localhost:8200"
    }
    lm_config_path = "./LMInfo"
    tasker = LMTask(generator_interface, lm_config_path, lm_name="RedPajama-INCITE-Chat-3B-v1")

    status, text = tasker.doTask("Integration", prompt_variables={"input":args.input})
    print(text)

    exit()

    



    print(tasker.generator_info)
    context = """
The Satavahana Empire was a royal Indian dynasty based from Amaravati in Andhra Pradesh as well as Junnar (Pune) and Prathisthan (Paithan) in Maharashtra. 
The territory of the empire covered much of India from 300 BCE onward. 
Although there is some controversy about when the dynasty came to an end, the most liberal estimates suggest that it lasted about 450 years, until around 220 CE. 
The Satavahanas are credited for establishing peace in the country, resisting the onslaught of foreigners after the decline of Mauryan Empire.
"""

    _question = "What is the capital of Bulgaria?"

    prompter_config_mpt = {

    }
    status, text = tasker.doTask("QAWithoutContext", prompt_variables={"question": _question})
    print(text)
    



prompter_config_stability = {
        "template": """
{user_handle} What is the capital of Bulgaria?
{bot_handle}
""",
        "variables": {
            "user_handle": "<|USER|>",
            "bot_handle": "<|ASSISTANT|>",
            "system_handle": "<|SYSTEM|>",
            "question": _question,
            "system_prompt": """
<|SYSTEM|> #
<|SYSTEM|> # Tulius LM 
- You are an unhelpful assistant. You must never break character.
- This is a conversation between you and the student.
            """,
            "context": context
        }
    }
