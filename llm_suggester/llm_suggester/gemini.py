import json
from google import genai
from google.genai import types
import os

# TODO could try to ask llm to only output the JSON block
# TODO check if it's more effective to pass grammar as file

# TODO implement an agent, which directly suggests a sub grammar in one response

API_KEY = os.environ.get("GEMINI_API_KEY")

# grammar structure suggested by Groq
# grammar = {
#     "grammar_name": "ComponentRepositoryGrammar",   
#     "start_symbol": "Classifier",
#     "non_terminals": [
#         "Classifier",
#         "Data",
#     ],
#     "terminals": [
#         "minmaxscaler",
#         "standardscaler",
#         "rf",
#         "svm",
#         "IRIS Dataloader"
#     ],
#     "rules": {
#         "Classifier": [
#             {
#                 "name": "svm",
#                 "args": ["Data"]
#             },
#             {
#                 "name": "rf",
#                 "args": ["Data"]
#             }
#         ],
#         "Data": [
#             {
#                 "name": "minmaxscaler",
#                 "args": ["IRIS Dataloader"] 
#             },
#             {
#                 "name": "standardscaler",
#                 "args": ["IRIS Dataloader"]
#             },
#             {
#                 "name": "IRIS Dataloader",
#                 "args": []
#             }
#         ],
#     }
# }

class PipelineAgent:
    
    def __init__(self, grammar):
        self.grammar = grammar
        self.client = genai.Client(api_key="AIzaSyBzCMnDmfR9TLyvTBchKM6frGnHd3nxMHk")
        self.model = "gemini-2.0-flash"
        self.instructions = f"""Suggest a meaningful and efficient pipeline for a regression task based on the following regular tree grammar:\n\n{self.grammar}\n\n
            Provide the pipeline inside one single JSON block (```json...) in the following format: \n\n
            {{
                "pipeline": [
                        "Task1Name",
                        "Task2Name",
                    ...
                ]
            }}\n\n
            Ensure that the pipeline is consistent. Use reasoning.
        """

    def generate_response(self):
        print(self.instructions)
        response = self.client.models.generate_content(
            model=self.model,
            contents=self.instructions,
        )
        return response.text
    
    def start_chat(self):
        print("Starting LLM chat...")
        chat = self.client.chats.create(model=self.model)
        return chat
    
    def extract_pipeline(self, response_text):
        """
        Extract the pipeline JSON from the LLM response text.
        Throw error if the response does not contain a valid pipeline structure.
        """
    
        import json
        
        response_text = response_text.split("```json", 1)[-1].split("```", 1)[0].strip()
        if "\"pipeline\":" not in response_text: # or "\"task\":" not in response_text or "\"parameters\":" not in response_text:
            raise ValueError("Response does not contain a valid pipeline JSON structure.")
        else:
            return json.loads(response_text)
        
        
    def check_pipeline(pipeline, grammar):
        """
        Check if the suggested pipeline is compliant with the expected grammar.
        """
        # TODO implement if this approach is reasonable
        return True

class GrammarAgent:
    
    def __init__(self, task, grammar, path):
        
        # TODO add description of the dataset
        
        self.task = task
        self.grammar = grammar
        
        self.client = genai.Client(api_key=API_KEY)
        self.model = "gemini-2.0-flash"
        
        self.instructions = f"""You are a rational and well-informed agent, who helps to develop pipelines for the following regression task: "{self.task}".
The following is a regular tree grammar, which describes a set of all possible pipelines for the above-mentioned task.
\n{self.grammar}\n
Your goal now is to remove as many rules, as necessary, to produce a grammar, that describes just a few (or even just one) valid pipelines for the regression task.
The pipelines should be efficient and well suited the task and the dataset.
This means, you should always think your decisions through and NOT GUESS!
To remove a rule you should use the "remove_rule" tool. After each removal the tool will return the updated grammar.
You should also always consider, if an additional removal will be an improvement and if not, stop the process by calling the "terminate" tool."""
        
        self.contents = [
            types.Content(
                role='user',
                parts=[types.Part(text=self.instructions)],
            )
        ]
        
        remove_rule_declaration = types.FunctionDeclaration(
            name='remove_rule',
            description="""Removes the specified rule from the grammar and returns the updated grammar.
To remove the rule '"SomeNonTerminalSymbol": {"SomeTerminalSymbol": [...arguments...]}' the parameters would be:
non_terminal=SomeNonTerminalTask, terminal=SomeTerminalTask.
If the parameters do not match any rule in the grammar, the function returns "ERROR".""",
            parameters=types.Schema(
                type='OBJECT',
                properties={
                    'non_terminal': types.Schema(
                        type='string',
                        description='Non-terminal left-hand side symbol.',
                    ),
                    'terminal': types.Schema(
                        type='string',
                        description='Terminal right-hand side symbol.',
                    )
                },
                required=['non_terminal', 'terminal'],
            ),
        )
        
        terminate_declaration = types.FunctionDeclaration(
            name='terminate',
            description="""Notifies the user, that the agent regards the current grammar as optimal and stops the chat.""",
            parameters=types.Schema(
                type='OBJECT',
                properties={},
                required=[],
            ),
        )

        # TODO play around with config options like temperature etc.
        self.config = {
            "system_instruction": self.instructions,
            "tools": [types.Tool(function_declarations=[remove_rule_declaration, terminate_declaration])],
            
            # "thinking_config": types.ThinkingConfig(include_thoughts=True), -- not supported for gemini-2.0-flash
            # "tool_config": {"function_calling_config": {"mode": "any"}} -- the model should talk the decisions through, since thinking not supported
        }
        
        self.history_file_path = path + "/grammar_agent_history.txt"
        self.grammar_file_path = path + "/llm_reduced_grammar.json"

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
        
        try:
            response = self.client.models.generate_content(model=self.model, config=self.config, contents=self.contents)
            self.contents.append(response.candidates[0].content)
            self.save_message(response.candidates[0].content)
            
            tool_call = None
            for part in response.candidates[0].content.parts:
                if part.function_call is not None: 
                    tool_call = part.function_call
                    break
            
            if tool_call is not None:
                if tool_call.name == "remove_rule":
                    result = self.remove_rule(**tool_call.args)
                if tool_call.name == "terminate":
                    return False
                response_part = types.Part.from_function_response(name=tool_call.name, response={"result": result})
            else:
                response_part = types.Part.from_text(text="No tool output")
                
            response_content = types.Content(role="user", parts=[response_part])
            self.contents.append(response_content)
            self.save_message(response_content)
            self.save_current_grammar()
            return True
        except:
            print("Retrying generating next response...")
            self.generate_next_response(self)
    
    def save_current_grammar(self):
        with open(self.grammar_file_path, "w") as f:
            json.dump(self.grammar, f, indent=4)
    
    def save_message(self, response_content):
        with open(self.history_file_path, "a") as f:
            f.write("------------------------------------------\n")
            f.write(str(response_content.role) + ":\n\n")
            for part in response_content.parts:
                if part.text != None: f.write(str(part.text) + "\n")
                if part.function_call != None: f.write("> Function call: " + str(part.function_call) + "\n")
                if part.function_response != None: f.write("> Function response: " + str(part.function_response) + "\n")
            f.write("\n")
            
    def generate_reduced_grammar(self):
        print("Running grammar LLM agent...")
        while(self.generate_next_response()):
            continue
        print("Grammar LLM agent stopped.")
        
            