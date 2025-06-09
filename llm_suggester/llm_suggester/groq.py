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
        
        self.instructions = f"""You are a rational and well-informed agent, who helps to develop pipelines for the following regression task: "{self.task}".
The following is a regular tree grammar, which describes a set of all possible pipelines for the above-mentioned task.
\n{self.grammar}\n
Your goal now is to remove as many rules, as necessary, to produce a grammar, that describes just a few (or even just one) valid pipelines for the regression task.
The pipelines should be efficient and well-suited for the task and the dataset.
This means, you should always think your decisions through and NOT GUESS!
To remove a rule you should use the "remove_rule" tool. After each removal the tool will return the updated grammar.
You should also always consider, if an additional removal will be an improvement and if not, stop the process by calling the "terminate" tool."""

        self.messages = [{
                    "role": "system",
                    "content": self.instructions,
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
                        self.grammar["non_terminals"].pop(non_terminal) # remove from non_terminals
                if "\"" + terminal + "\"" not in str(self.grammar["rules"]): # if terminal no longer appears in rules
                    print(self.grammar["terminals"])
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
        )
        response = chat_completion.choices[0].message
        print("RESPONSE")
        print(response)
        self.messages.append(response)
        tool_calls = response.tool_calls
        if tool_calls:
            self.messages.append(response)
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                print("TOOL CALL:", function_name, tool_call)
                if function_name == "remove_rule":
                    print("HERE")
                    result = self.remove_rule(**tool_call.function.arguments)
                elif function_name == "terminate":
                    return False
                self.messages.append({
                    "tool_call_id": tool_call.id, 
                    "role": "tool",
                    "name": function_name,
                    "content": result,
                })
        self.save_response(response)
        self.save_current_grammar()
        return True
    
    def save_current_grammar(self):
        with open(self.grammar_file_path, "w") as f:
            json.dump(self.grammar, f, indent=4)
    
    def save_response(self, response_content):
        print("RESPONSE:", response_content)
        with open(self.history_file_path, "a") as f:
            f.write("------------------------------------------\n")
            f.write(str(response_content.role) + ":\n\n")
            if response_content != None: f.write(str(response_content) + "\n")
            f.write("\n")