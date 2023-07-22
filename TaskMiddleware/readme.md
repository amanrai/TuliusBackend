Prompting

There are four different system level variables that the prompter will try to insert into the final string. 
{user_handle} = This is how the user is referred to in a conversation;l usually something like <human>: or User:
{bot_handle} = This is how the bot is reffered to in a conversation; usually something like <bot>: or Assistant:
{system_handle} = This is how the system is referred to in a conversaiton; usually something like <|SYSTEM|> or System
{instruction guideline} - The repeatable string that is given to the LM for instructional tasks. For example, with Alpaca models we usually start with saying something like "Below is an instruction..."

Additionally, there are four standard runtime variables that you can use:
{question} - What to ask the LM
{context} - The context to provide to the LM
{instruction} - Any specific instruction to provide the LM
{input} - Any other inputs

NOTE: Unless the template explicitly supports the above variables, they will NOT be inserted (true of both system and runtime variables)

All served templates are to be stored in the Prompts subfolder with the filename <promptTemplateName>.txt

There is a custom template function that allows you to send a template and variables to insert into the template. 
This function does not use the existing system level variables by default. If you want to use them, pass insert_system_variables = True