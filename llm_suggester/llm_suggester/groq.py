import json
from groq import Groq

import os

API_KEY = os.environ.get("GROQ_API_KEY")

class GrammarAgent:
    
    def __init__(self, task, grammar, path):
        self.task = task
        self.grammar = grammar
        self.path = path
        self.model = "llama3-8b-8192"
        self.client = Groq(api_key=API_KEY)
        
        self.instructions = f"""You are a rational and well-informed agent, who helps to develop pipelines for regression tasks.
You will be provided with a regular tree grammar, which describes a set of all possible pipelines for the a task, that will be specified by user.
Your goal is to remove as many rules, as necessary, to get to a grammar, that describes just a few (or even just one) valid pipelines for the regression task.
The pipelines should be efficient and well-suited for the task and the dataset.
This means, you should always think your decisions through and NOT GUESS!
To remove a rule you should use the "remove_rule" tool. After each removal the tool will return the updated grammar.
You should also always consider, if an additional removal will be an improvement and if not, stop the process by calling the "terminate" tool. Feel free to ."""

        self.first_user_message = f"Task: {self.task}\n\nGrammar:\n{self.grammar}"

        self.messages = [{
                    "role": "system",
                    "content": self.instructions,
                },
                {
                    "role": "user",
                    "content": self.first_user_message         
                }]
        
        self.tools = [{
            "type": "function",
            "function": {
                "name": "remove_rule",
                "description": """Removes the specified rule from the grammar and returns the updated grammar.
To remove the rule '"SomeNonTerminalSymbol": {"SomeTerminalSymbol": [...arguments...]}' the parameters would be:
non_terminal=SomeNonTerminalTask, terminal=SomeTerminalTask.
If the parameters do not match any rule in the grammar, the function returns "ERROR".""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "non_terminal": {
                            "type": "string",
                            "description": "Non-terminal left-hand side symbol."
                        },
                        "terminal": {
                            "type": "string",
                            "description": "Terminal right-hand side symbol."
                        }
                    },
                    "required": ["non_terminal", "terminal"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "terminate",
                "description": """Notifies the user, that the agent regards the current grammar as optimal and stops the chat.""",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }]
        
        self.history_file_path = path + "/grammar_agent_history.txt"
        self.grammar_file_path = path + "/suggested_grammar.json"
    
    # remove_rule tool
    def remove_rule(self, non_terminal, terminal):
        for rule in self.grammar["rules"]:
            if rule == non_terminal:
                self.grammar["rules"][rule].pop(terminal)
                if self.grammar["rules"][rule] == {}: # if the terminal was the last for this rule
                    self.grammar["rules"].pop(rule) # remove whole rule
                    if "\"" + non_terminal + "\"" not in str(self.grammar["rules"]): # if non_terminal no longer appears in rules
                        self.grammar["non_terminals"].remove(non_terminal) # remove from non_terminals
                if "\"" + terminal + "\"" not in str(self.grammar["rules"]): # if terminal no longer appears in rules
                    self.grammar["terminals"].remove(terminal) # remove from terminals
                return str(self.grammar)
        return "ERROR"
    
    # terminate tool
    def terminate(self):
        print("Grammar agent terminated.")
    
    def generate_next_response(self):
        
        chat_completion = self.client.chat.completions.create(
            messages=self.messages,
            model=self.model,
            stream=False,
            tools=self.tools,
            tool_choice="auto"
        )
        tool_calls = chat_completion.choices[0].message.tool_calls
        llm_message = {
            "role": "assistant",
            "content": chat_completion.choices[0].message.content,
        }
        self.messages.append(llm_message)
        self.save_message(llm_message)
        if tool_calls:
            llm_message["tool_calls"] = tool_calls
            tool_call = tool_calls[0]
            function_name = tool_call.function.name
            if function_name == "remove_rule":
                result = self.remove_rule(**json.loads(tool_call.function.arguments))
                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": result
                }
                self.messages.append(tool_message)
                self.save_message(tool_message)
            elif function_name == "terminate":
                return False
        return True

    def save_current_grammar(self):
        with open(self.grammar_file_path, "w") as f:
            json.dump(self.grammar, f, indent=4)
    
    def save_message(self, message):
        with open(self.history_file_path, "a") as f:
            f.write("------------------------------------------\n")
            f.write(str(message["role"]) + ":\n\n")
            if message["content"]: f.write(str(message["content"]) + "\n")
            if "tool_calls" in message: f.write(str(message["tool_calls"]) + "\n")
            f.write("\n")