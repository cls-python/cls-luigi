from ollama import chat


MODEL_NAME = "llama3.2"


class DirectGrammarAgent:

    def __init__(self, task, grammar, path):
        
        self.task = task
        self.grammar = grammar
        
        self.model = MODEL_NAME
        
        self.history_file_path = path + "/grammar_agent_history.txt"
        self.grammar_file_path = path + "/llm_reduced_grammar.json"
        
        self.instructions = f"""You are a helpful and rational agent, and your goal is to create a pipeline for the following machine learning task: "{self.task}".\n
The following is a regular tree grammar, which defines the pipeline tasks and the rules for combining them, to build all possible pipelines for the above-mentioned task.\n{self.grammar}\n
Your goal is to reduce the grammar, so that it produces only one valid pipeline, which you regard as the most well-suited for the task and the dataset. To suggest the modified grammar you should use the "suggest_grammar" tool.
All your choices should be well thought out, so do not hesitate to explain your thinking process in the response."""

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
        
    def suggest_grammar(self, grammar):
        """Suggests the modified grammar to the user."""
        print(f"Suggested grammar: {grammar}")
        return {'grammar': grammar}

    def generate_response(self, input_text):
        self.messages.append({
            'role': 'user',
            'content': input_text
        })
        response = chat(model=self.model, messages=self.messages)
        self.messages.append({
            'role': 'assistant',
            'content': response
        })
        print(f"Response from model: {response}")
        return {'role': 'assistant', 'content': response}

