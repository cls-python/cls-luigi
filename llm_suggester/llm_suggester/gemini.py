import json
from google import genai
from google.genai import types
import os

# TODO could try to ask llm to only output the JSON block

API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.0-flash"

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
    
    def __init__(self, task, grammar, path):
        
        self.task = task
        self.grammar = grammar
        
        self.example_pipeline = """{
    "nodes": [
        {
            "name": "JSONLoader",
        },
        {
            "name": "MissingValueImputation",
            "inputs": ["JSONLoader"]
        },
        {
            "name": "OneHotEncoding",
            "inputs": ["MissingValueImputation"]
        },
        {
            "name": "StandardScaler",
            "inputs": ["MissingValueImputation"]
        },
        {
            "name": "RF",
            "inputs": ["OneHotEncoding", "StandardScaler"]
        },
        {
            "name": "LR",
            "inputs": ["StandardScaler"]
        },
        {
            "name": "VotingClassifier",
            "inputs": ["RF", "LR"]
        },
        {
            "name": "Eval",
            "inputs": ["VotingClassifier"]
        }
    ]
}"""
        
        self.client = genai.Client(api_key=API_KEY)
        self.model = MODEL_NAME
        
        self.history_file_path = path + "/grammar_agent_history.txt"
        self.pipeline_file_path = path + "/llm_suggested_pipeline.json"
        
        self.instructions = f"""You are a helpful agent, who will help to develop a pipeline for the following regression task: "{self.task}".

The following is a regular tree grammar, which defines the pipeline tasks and rules for combining them, to build all possible pipelines for the above-mentioned task.

{self.grammar}

Based on that you will now produce one pipeline, which is the most well-suited for the task and the dataset. To suggest a pipeline you will use the "suggest_pipeline" tool.

You will suggest the pipeline in JSON format. Here is an example for a pipeline:

{self.example_pipeline}

Each node is a terminal and has a name and inputs from other nodes.

The pipeline is not linear in general. You will produce a pipeline with branching and merging paths, if it is required by the task and data.

You will think all your decisions through by using thorough reasoning, to produce the best possible result.

Now start!"""
        
        self.contents = [
                types.Content(
                    role='user',
                    parts=[types.Part(text=self.instructions)],
                )
            ]

        suggest_pipeline_declaration = types.FunctionDeclaration(
            name='suggest_pipeline',
            description='This tool is used to suggest a pipeline.',
            parameters=types.Schema(
                type='OBJECT',
                properties={
                    'pipeline': types.Schema(
                        type='string',
                        description='Suggested pipeline in the correct format',
                    )
                },
                required=['pipeline'],
            ),
        )
            
        self.config = {
            # "system_instruction": self.instructions,
            "tools": [types.Tool(function_declarations=[suggest_pipeline_declaration])],
            # "thinking_config": types.ThinkingConfig(include_thoughts=True), -- not supported for gemini-2.0-flash
            # "tool_config": {"function_calling_config": {"mode": "any"}} -- the model should talk the decisions through, since thinking not supported
        }
    
    
    def suggest_pipeline(self, pipeline):
    
        print("Suggested pipeline:", pipeline)
        pipeline = pipeline.replace("'", "\"") # replace single quotes with double quotes to make it valid JSON
        # TODO check if pipeline is valid via cls(?) (or do it outside the agent class)
        self.save_pipeline(pipeline)
        return True
    
    def generate_pipeline(self):
        
        try:
            response = self.client.models.generate_content(model=self.model, config=self.config, contents=self.contents)
            self.contents.append(response.candidates[0].content)
            self.save_message(response.candidates[0].content)            
            tool_call = None
            for part in response.candidates[0].content.parts:
                if part.function_call is not None: 
                    tool_call = part.function_call
                    break
            if tool_call is not None and tool_call.name == "suggest_pipeline":
                result = self.suggest_pipeline(**tool_call.args)
                response_part = types.Part.from_function_response(name=tool_call.name, response={"result": result})
            else:
                response_part = types.Part.from_text(text="No tool output")
                print("Pipeline agent stopped without producing a pipeline.")
            response_content = types.Content(role="user", parts=[response_part])
            self.contents.append(response_content)
            self.save_message(response_content)
        except Exception as e:
            print("Error occurred while generating pipeline:", e)
            print("Retrying generating next response...")
            self.generate_pipeline()
    
    def save_pipeline(self, new_pipeline):
        with open(self.pipeline_file_path, "w") as f:
            json.dump(json.loads(new_pipeline), f, indent=4)
        print("Suggested Pipeline saved to", self.pipeline_file_path)
            
    def save_message(self, message):
        with open(self.history_file_path, "a") as f:
            f.write("------------------------------------------\n")
            f.write(str(message.role) + ":\n\n")
            for part in message.parts:
                if part.text != None: f.write(str(part.text) + "\n")
                if part.function_call != None: f.write("> Function call: " + str(part.function_call) + "\n")
                if part.function_response != None: f.write("> Function response: " + str(part.function_response) + "\n")
            f.write("\n")

    
    
    # def extract_pipeline(self, response_text):
    #     """
    #     Extract the pipeline JSON from the LLM response text.
    #     Throw error if the response does not contain a valid pipeline structure.
    #     """
    
    #     import json
        
    #     response_text = response_text.split("```json", 1)[-1].split("```", 1)[0].strip()
    #     if "\"pipeline\":" not in response_text: # or "\"task\":" not in response_text or "\"parameters\":" not in response_text:
    #         raise ValueError("Response does not contain a valid pipeline JSON structure.")
    #     else:
    #         return json.loads(response_text)
        
        
    # def check_pipeline(pipeline, grammar):
    #     """
    #     Check if the suggested pipeline is compliant with the expected grammar.
    #     """
    #     # TODO implement if this approach is reasonable
    #     return True

class DirectGrammarAgent:
    
    def __init__(self, task, grammar, path):
        
        self.task = task
        self.grammar = grammar
        
        self.client = genai.Client(api_key=API_KEY)
        self.model = MODEL_NAME
        
        self.history_file_path = path + "/grammar_agent_history.txt"
        self.grammar_file_path = path + "/llm_reduced_grammar.json"
        
        self.instructions = f"""You are a helpful and rational agent, and your goal is to create a pipeline for the following machine learning task: "{self.task}".\n
The following is a regular tree grammar, which defines the pipeline tasks and the rules for combining them, to build all possible pipelines for the above-mentioned task.\n{self.grammar}\n
Your goal is to reduce the grammar, so that it produces only one valid pipeline, which you regard as the most well-suited for the task and the dataset. To suggest the modified grammar you should use the "suggest_grammar" tool.
All your choices should be well thought out, so do not hesitate to explain your thinking process in the response."""

        self.contents = [
            types.Content(
                role='user',
                parts=[types.Part(text=self.instructions)],
            )
        ]
        
        self.save_message(self.contents[0])

        suggest_grammar_declaration = types.FunctionDeclaration(
            name='suggest_grammar',
            description="""Passes the suggested grammar to the user. """,
            parameters=types.Schema(
                type='OBJECT',
                properties={
                    'grammar': types.Schema(
                        type='string',
                        description='The regular tree grammar in string format.',
                    ),
                },
                required=['grammar'],
            ),
        )

        # TODO play around with config options like temperature etc.
        self.config = {
            # "system_instruction": self.instructions,
            "tools": [types.Tool(function_declarations=[suggest_grammar_declaration])],
            # "thinking_config": types.ThinkingConfig(include_thoughts=True), -- not supported for gemini-2.0-flash
            # "tool_config": {"function_calling_config": {"mode": "any"}} -- the model should talk the decisions through, since thinking not supported
        }
        
    # suggest_grammar tool
    # takes the suggested regular tree grammar
    # if it's valid, saves it to llm_reduced_grammar.json and returns True, else returns False
    def suggest_grammar(self, grammar):
        print("Suggested grammar:", grammar)
        grammar = grammar.replace("'", "\"") # replace single quotes with double quotes to make it valid JSON
        # TODO (?) check if grammar produces valid pipeline via cls(?)
        return grammar
        
    def generate_reduced_grammar(self):
        try:
            response = self.client.models.generate_content(model=self.model, config=self.config, contents=self.contents)
            self.contents.append(response.candidates[0].content)
            self.save_message(response.candidates[0].content)
            
            tool_call = None
            for part in response.candidates[0].content.parts:
                if part.function_call is not None: 
                    tool_call = part.function_call
                    break
            if tool_call is not None and tool_call.name == "suggest_grammar":
                grammar_str = self.suggest_grammar(**tool_call.args)
                response_part = types.Part.from_function_response(name=tool_call.name, response={"result": grammar_str})
                grammar = self.grammar_str_to_json(grammar_str)
                self.save_grammar(grammar)
            else:
                grammar = None
                response_part = types.Part.from_text(text="No tool output")
                print("Direct grammar agent stopped without producing a grammar.")
            response_content = types.Content(role="user", parts=[response_part])
            self.contents.append(response_content)
            self.save_message(response_content)
            return grammar
        except Exception as e:
            print("Error occurred while generating reduced grammar:", e)
            print("Retrying generating next response...")
            self.save_retry_message()
            return self.generate_reduced_grammar()
    
    def save_grammar(self, new_grammar):
        with open(self.grammar_file_path, "w") as f:
            json.dump(new_grammar, f, indent=4)
        print("Grammar saved to", self.grammar_file_path)
            
    def grammar_str_to_json(self, grammar_str):
        try:
            return json.loads(grammar_str)
        except json.JSONDecodeError as e:
            print("Error decoding JSON:", e)
            return None
            
    def save_message(self, message):
        with open(self.history_file_path, "a") as f:
            f.write("------------------------------------------\n")
            f.write(str(message.role) + ":\n\n")
            for part in message.parts:
                if part.text != None: f.write(str(part.text) + "\n")
                if part.function_call != None: f.write("> Function call: " + str(part.function_call) + "\n")
                if part.function_response != None: f.write("> Function response: " + str(part.function_response) + "\n")
            f.write("\n")
    
    def save_retry_message(self):
        with open(self.history_file_path, "a") as f:
            f.write("------------------------------------------\n")
            f.write("")
            f.write("Retrying generating next response...\n\n")
            f.write("\n")

class IterativeGrammarAgent:
    
    def __init__(self, task, grammar, path):
        
        # TODO add description of the dataset
        
        self.task = task
        self.grammar = grammar
        
        self.client = genai.Client(api_key=API_KEY)
        self.model = MODEL_NAME
        
        # TODO just one pipeline
        
        self.instructions = f"""You are a rational and well-informed agent, who helps to develop pipelines for the following task: "{self.task}".
The following is a regular tree grammar, which describes a set of all possible pipelines for the above-mentioned task.
\n{self.grammar}\n
Your goal now is to remove as many rules, as necessary, to produce a grammar, that describes just one valid pipeline for the regression task.
The pipeline should be efficient and well suited for the task and the dataset.
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
        
            
class IterativeFeedbackGrammarAgent:
    
    def __init__(self, task, grammar, path):
        
        self.task = task
        self.grammar = grammar
        
        self.client = genai.Client(api_key=API_KEY)
        self.model = MODEL_NAME
        
        self.history_file_path = path + "/grammar_agent_history.txt"
        self.grammar_file_path = path + "/llm_reduced_grammar.json"
        
        self.instructions = f"""You are a helpful and rational agent, and you create a pipeline for the following machine learning task: "{self.task}".\n
The following is a regular tree grammar, which defines the pipeline tasks and the rules for combining them, to build all possible pipelines for the above-mentioned task.\n{self.grammar}\n
You reduce the grammar, so that it produces only one valid pipeline, which you regard as well-suited for the task and the dataset. To suggest the modified grammar you use the "suggest_grammar" tool. After you suggest the grammar, the tool will return the performance score of the pipeline, which is produced by the grammar. You then suggest a new grammar to improve the next score. You repeat this process until you see no way to further improve any of the suggested grammars and stop the process by calling the "terminate" tool.\n
All your choices should be well thought out, so you should explain your thinking process in each response."""

        self.contents = [
            types.Content(
                role='user',
                parts=[types.Part(text=self.instructions)],
            )
        ]
        
        self.save_message(self.contents[0])

        suggest_grammar_declaration = types.FunctionDeclaration(
            name='suggest_grammar',
            description="""Passes the suggested grammar to the user. """,
            parameters=types.Schema(
                type='OBJECT',
                properties={
                    'grammar': types.Schema(
                        type='string',
                        description='The regular tree grammar in string format.',
                    ),
                },
                required=['grammar'],
            ),
        )
        
        terminate_declaration = types.FunctionDeclaration(
            name='terminate',
            description="""Notifies the user, that the agent sees no way to further improve the grammar and stops the chat.""",
            parameters=types.Schema(
                type='OBJECT',
                properties={},
                required=[],
            ),
        )

        # TODO play around with config options like temperature etc.
        self.config = {
            # "system_instruction": self.instructions,
            "tools": [types.Tool(function_declarations=[suggest_grammar_declaration, terminate_declaration])],
            # "thinking_config": types.ThinkingConfig(include_thoughts=True), -- not supported for gemini-2.0-flash
            # "tool_config": {"function_calling_config": {"mode": "any"}} -- the model should talk the decisions through, since thinking not supported
        }
        
    # suggest_grammar tool
    # takes the suggested regular tree grammar
    # if it's valid, saves it to llm_reduced_grammar.json and returns True, else returns False
    def suggest_grammar(self, grammar):
        print("Suggested grammar:", grammar)
        grammar = grammar.replace("'", "\"") # replace single quotes with double quotes to make it valid JSON
        # TODO (?) check if grammar produces valid pipeline via cls(?)
        return grammar
        
    def terminate(self):
        print("Grammar agent terminated.")
        return True
        
    def generate_reduced_grammar(self):
        try:
            response = self.client.models.generate_content(model=self.model, config=self.config, contents=self.contents)
            self.contents.append(response.candidates[0].content)
            self.save_message(response.candidates[0].content)
            
            tool_call = None
            for part in response.candidates[0].content.parts:
                if part.function_call is not None: 
                    tool_call = part.function_call
                    break
            if tool_call is not None and tool_call.name == "suggest_grammar":
                grammar_str = self.suggest_grammar(**tool_call.args)
                response_part = types.Part.from_function_response(name=tool_call.name, response={"result": grammar_str})
                grammar = self.grammar_str_to_json(grammar_str)
                self.save_grammar(grammar)
            else:
                grammar = None
                response_part = types.Part.from_text(text="No tool output")
                print("Iterative feedback grammar agent stopped without producing a grammar.")
            response_content = types.Content(role="user", parts=[response_part])
            self.contents.append(response_content)
            self.save_message(response_content)
            return grammar
        except Exception as e:
            print("Error occurred while generating reduced grammar:", e)
            print("Retrying generating next response...")
            self.save_retry_message()
            return self.generate_reduced_grammar()
    
    def save_grammar(self, new_grammar):
        with open(self.grammar_file_path, "w") as f:
            json.dump(new_grammar, f, indent=4)
        print("Grammar saved to", self.grammar_file_path)
            
    def grammar_str_to_json(self, grammar_str):
        try:
            return json.loads(grammar_str)
        except json.JSONDecodeError as e:
            print("Error decoding JSON:", e)
            return None
            
    def save_message(self, message):
        with open(self.history_file_path, "a") as f:
            f.write("------------------------------------------\n")
            f.write(str(message.role) + ":\n\n")
            for part in message.parts:
                if part.text != None: f.write(str(part.text) + "\n")
                if part.function_call != None: f.write("> Function call: " + str(part.function_call) + "\n")
                if part.function_response != None: f.write("> Function response: " + str(part.function_response) + "\n")
            f.write("\n")
    
    def save_retry_message(self):
        with open(self.history_file_path, "a") as f:
            f.write("------------------------------------------\n")
            f.write("")
            f.write("Retrying generating next response...\n\n")
            f.write("\n")