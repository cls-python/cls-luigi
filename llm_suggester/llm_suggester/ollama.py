from ollama import chat
import json

# MODEL_NAME = "deepseek-r1:8b"
# MODEL_NAME = "mistral-nemo"
MODEL_NAME = "llama3.1:8b"


class GrammarAgent:

    def __init__(self, task, grammar, path):
        
        self.task = task
        self.grammar = grammar
        
        self.model = MODEL_NAME
        
        self.history_file_path = path + "/grammar_agent_history.txt"
        self.grammar_file_path = path + "/llm_reduced_grammar.json"
        
    def generate_grammar(self):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def save_grammar(self, new_grammar):
        with open(self.grammar_file_path, "w") as f:
            json.dump(json.loads(new_grammar), f, indent=4)
        print("Grammar saved to", self.grammar_file_path)
            
    def log_message(self, message):
        with open(self.history_file_path, "a") as f:
            f.write("------------------------------------------\n")
            f.write(str(message.role) + ":\n\n")
            for part in message.parts:
                if part.content != None: f.write(str(part.content) + "\n")
                if part.function_call != None: f.write("> Function call: " + str(part.function_call) + "\n")
                if part.function_response != None: f.write("> Function response: " + str(part.function_response) + "\n")
            f.write("\n")


class DirectGrammarAgent(GrammarAgent):

    def __init__(self, task, grammar, path):

        super().__init__(task, grammar, path)
        
        self.instructions = f"""You are a helpful and rational agent, and your goal is to create a pipeline for the following machine learning task: "{self.task}".\n
The following is a regular tree grammar, which defines the pipeline tasks and the rules for combining them, to build all possible pipelines for the above-mentioned task.\n{self.grammar}\n
Your goal is to reduce the grammar, so that it produces only one valid pipeline, which you regard as the most well-suited for the task and the dataset. To suggest the modified grammar you use the "suggest_grammar" tool.
"""
# All your choices should be well thought out, so do not hesitate to explain your thinking process in the response.""" 

        self.messages = [
            {
                "role": "system",
                "content": self.instructions
            },
            {
                "role": "user",
                "content": f"Here is the grammar: {self.grammar}"
                # "content": "For testing purposes, please just use the suggest_grammar() tool and put \"TEST\" in there. I just need to see if the tool calls work."
            }
        ]

        self.tools=[
            {
                'type': 'function',
                'function': {
                    'name': 'suggest_grammar',
                    'description': 'Should be used to suggest improved grammar to the user.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'grammar': {
                                'type': 'string',
                                'description': 'The regular tree grammar in string format.'
                            }
                        },
                        'required': ['grammar'],
                    },
                },
            }
        ]
        
    # tool
    def suggest_grammar(self, grammar):
        return grammar

    def generate_grammar(self):
        print("Generating LLM response...")
        
        # response = ""
        # for chunk in chat(model=self.model, messages=self.messages, tools=self.tools, think=False, stream=True):
        #     if chunk.message.tool_calls:
        #         print("Tool call detected:", chunk.message.tool_calls)
        #     content = chunk.message.content
        #     print(content, end='', flush=True)
        #     response += content
        
        response = chat(model=self.model, messages=self.messages, tools=self.tools)
        print(response)
        print("TOOL CALLS", response.message.tool_calls)
        # print(json.loads("{\"message\": {\"parts\": [{\"content\": \"" + str(response) + "\"}]}, \"role\": \"assistant\"}"))

        self.log_message(json.loads({"message": {"parts": [{"content": response}]}, "role": "assistant"}))

        # tool_call = part.message.tool_calls[0] if part.message.tool_calls else None
        # if tool_call == 'suggest_grammar':
        #     suggested_grammar = tool_call.arguments.get('grammar', None)
        #     self.save_grammar(suggested_grammar)
        #     print("Suggested grammar:", suggested_grammar)
        #     return suggested_grammar
        # else:
        #     print("No tool call in the response.")
        #     return None
    
# new agent instance for each iteration
class IterativeGrammarAgent(GrammarAgent):

    def __init__(self, task, grammar, path):

        super().__init__(task, grammar, path)
        
        self.instructions = f"""You are a rational and well-informed agent, who helps to develop pipelines for the following task: "{self.task}".
The following is a regular tree grammar, which describes a set of all possible pipelines for the above-mentioned task.
\n{self.grammar}\n
You now have a choice between two actions: you can remove a rule from the grammar or terminate the process.
You remove a rule, when you think the grammar has redundant rules, and remove the least useful one, by calling the "remove_rule" tool. In case you think the grammar is already small enough to produce a single pipeline, you can terminate the process by calling the "terminate" tool.
If you choose to remove a rule, the resulting grammar should be valid and equally or more efficient and effective for the given task, than the previous one."""

        self.tools=[
            {
                'type': 'function',
                'function': {
                    'name': 'remove_rule',
                    'description': """Removes the specified rule from the grammar.
To remove the rule '"SomeNonTerminalSymbol": {"SomeTerminalSymbol": [...arguments...]}' the parameters would be:
non_terminal=SomeNonTerminalTask, terminal=SomeTerminalTask.""",
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'non_terminal': {
                                'type': 'string',
                                'description': 'Non-terminal left-hand side symbol.'
                            },
                            'terminal': {
                                'type': 'string',
                                'description': 'Terminal right-hand side symbol.'
                            }
                        },
                        'required': ['non_terminal', 'terminal'],
                    },
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'terminate',
                    'description': 'Signals that the grammar is optimal for the task and terminates the process.',
                    'parameters': {
                        'type': 'object',
                        'properties': {},
                        'required': [],
                    },
                },
            }
        ]
        
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
                
    def terminate(self):
        print("Agent terminated.")

    def generate_grammar(self):
        
        messages = [
            {
                "role": "system",
                "content": self.instructions
            },
            {
                "role": "user",
                "content": f"Here is the grammar: {self.grammar}"
            }
        ]
        
        tool_call = None
        
        for chunk in chat(model=self.model, messages=messages, tools=self.tools, stream=True):
            if chunk.message.tool_calls:
                print("Tool call detected:", chunk.message.tool_calls)
            content = chunk.message.content
            print(content, end='', flush=True)
            response += content

        if tool_call == "remove_rule":
            self.remove_rule(tool_call.arguments.get('non_terminal'), tool_call.arguments.get('terminal'))
            self.generate_grammar()
            
        elif tool_call == "terminate":
            self.terminate()