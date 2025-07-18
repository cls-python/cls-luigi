from ollama import chat
import json

MODEL_NAME = "deepseek-r1:8b"


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
Your goal is to reduce the grammar, so that it produces only one valid pipeline, which you regard as the most well-suited for the task and the dataset. To suggest the modified grammar you should use the "suggest_grammar" tool.
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
            }
        ]

        self.tools=[
            {
                'type': 'function',
                'function': {
                    'name': 'suggest_grammar',
                    'description': 'Passes the suggested grammar to the user.',
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
        
        response = ""
        for part in chat(model=self.model, messages=self.messages, tools=self.tools, stream=True):
            if part.message.tool_calls:
                print("Tool call detected:", part.message.tool_calls)
            content = part.message.content
            print(content, end='', flush=True)
            response += content

        print(json.loads("{\"message\": {\"parts\": [{\"content\": \"" + str(response) + "\"}]}, \"role\": \"assistant\"}"))

        self.log_message(json.loads({"message": {"parts": [{"content": str(response)}]}, "role": "assistant"}))

        tool_call = part.message.tool_calls[0] if part.message.tool_calls else None
        if tool_call == 'suggest_grammar':
            suggested_grammar = tool_call.arguments.get('grammar', None)
            self.save_grammar(suggested_grammar)
            print("Suggested grammar:", suggested_grammar)
            return suggested_grammar
        else:
            print("No tool call in the response.")
            return None
    
