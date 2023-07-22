#Aman Rai, July 2023
import os

class LMPrompter:

    def __init__(self, user_handle = None, bot_handle = None, system_handle = None, instruction_guideline = None, prompt_template_folder = "./Prompts"):
        self.user_handle = user_handle
        self.bot_handle = bot_handle
        self.system_handle = system_handle
        self.instruction_guideline = instruction_guideline
        self.prompt_template_folder = prompt_template_folder

        files_ = os.listdir(self.prompt_template_folder)
        self.prompt_choices = {}
        for file in files_:
            if (".txt" in file and not("custom" == file.lower())):
                with open(os.path.join(self.prompt_template_folder, file), "r") as f:
                    self.prompt_choices[file[:-4]] = f.read()

        print("These are your prompt Choices: ", self.prompt_choices.keys())

    def prompt(self, prompt_type, question = None, context = None, instruction = None, input=None):
        
        context = "" if context == None else context
        
        return self.prompt_choices[prompt_type].format(
            user_handle = self.user_handle,
            bot_handle = self.bot_handle,
            system_handle = self.system_handle,
            instruction_guideline = self.instruction_guideline,
            question = question,
            context = context,
            instruction = instruction,
            input = input
        )
    
    def runtimeCustomTemplate(self, prompt_template, variables, insert_system_variables=False):
        if (type(variables) == dict):
            if (insert_system_variables):
                p_template = prompt_template.format(
                    user_handle = self.user_handle,
                    bot_handle = self.bot_handle,
                    system_handle = self.system_handle,
                    instruction_guideline = self.instruction_guideline,
                    **variables
                )
                return p_template
            return prompt_template.format(**variables)
        else:
            return ""
