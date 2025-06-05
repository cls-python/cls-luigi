import json
from groq import Groq


class GrammarAgent:
    
    def __init__(self, task, grammar, path):
        self.task = task
        self.grammar = grammar
        self.path = path
        self.model = "llama3-8b-8192"
        self.client = Groq(api_key="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",)
        
        self.instructions = f"""You are a rational and well-informed agent, who helps to develop pipelines for the following regression task: "{self.task}".
The following is a regular tree grammar, which describes a set of all possible pipelines for the above-mentioned task.
\n{self.grammar}\n
Your goal now is to remove as many rules, as necessary, to produce a grammar, that describes just a few (or even just one) valid pipelines for the regression task.
The pipelines should be efficient and well suited the task and the dataset.
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
To remove the rule '"SomeNonTerminalTask": {"SomeTerminalTask": ["SomeOtherNonTerminalTask"]}' the arguments would be:
non_terminal_left=SomeNonTerminalTask, terminal_right=SomeTerminalTask, non_terminal_right=SomeOtherNonTerminalTask.
To remove the rule '"SomeNonTerminalTask": {"SomeTerminalTask": [...and any non-terminal in here...]}' the arguments would be:
non_terminal_left=SomeNonTerminalTask, terminal_right=SomeTerminalTask, non_terminal_right=None (meaning every non-terminal inside the terminal will be removed with the terminal).
If the arguments do not match any rule in the grammar, the function returns "ERROR".""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "non_terminal_left": {
                            "type": "string",
                            "description": "Non-terminal left-hand side symbol."
                        },
                        "terminal_right": {
                            "type": "string",
                            "description": "Terminal right-hand side symbol."
                        },
                        "non_terminal_right": {
                            "type": "string",
                            "description": "Non-terminal right-hand side symbol. Only required, if the rule contains a right-hand side non-terminal.",
                        }
                    },
                    "required": ["non_terminal_left", "terminal_right"]
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
    def remove_rule(self, non_terminal_left, terminal_right, non_terminal_right=None):
        # TODO remove symbols from terminals and non_terminals also, if they do not occur in rules anymore
        for rule in self.grammar["rules"]:
            if rule == non_terminal_left:
                if non_terminal_right is None:
                    self.grammar["rules"][rule].pop(terminal_right)
                    if self.grammar["rules"][rule] == {}: # if the terminal was the last for this rule
                        self.grammar["rules"].pop(rule) # remove whole rule
                else:
                    self.grammar["rules"][rule][terminal_right].remove(non_terminal_right)
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
        self.messages.append(response)
        tool_calls = response.tool_calls
        if tool_calls:
            self.messages.append(response)
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                print("TOOL CALL:", function_name, tool_call.args)
                if function_name == "remove_rule":
                    result = self.remove_rule(**tool_call.args)
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