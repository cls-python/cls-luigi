import json
from google import genai
from google.genai import types

# TODO could try to ask llm to only output the JSON block
# TODO check if it's better to pass grammar as file

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
    
    def __init__(self, task, grammar):
        
        # TODO add task description to prompt

        self.task = task
        self.grammar = grammar
        
        self.client = genai.Client(api_key="AIzaSyBzCMnDmfR9TLyvTBchKM6frGnHd3nxMHk")
        self.model = "gemini-2.0-flash"
        
        self.instructions = f"""
        You are a rational and well-informed agent, who helps to develop pipelines for the following regression task: "{self.task}".
        The following is a regular tree grammar, which describes a set of all possible pipelines for the above-mentioned task.
        \n{self.grammar}\n
        You are now going to iteratively chose which rule to eliminate, until the grammar can only produce one valid pipeline.
        To remove a rule you should use the remove_rule tool. After each removal the tool will return the updated grammar.
        Your goal is to remove as many rules, as necessary, to produce a grammar, that describes just a few (or even just one) meaningful and efficient pipelines for the regression task.
        This means, you should always think your decisions through and NOT guess! You should also always check, if an additional removal will be an improvement and if not, stop by calling the terminate tool.
        """
        
        self.contents = [
            types.Content(
                role='user',
                parts=[types.Part(text=self.instructions)],
            )
        ]
        
        remove_rule_declaration = types.FunctionDeclaration(
            name='remove_rule',
            description="""Removes the specified rule from the grammar and returns the updated grammar.
            To remove the rule '"SomeNonTerminalTask": {"SomeTerminalTask": ["SomeOtherNonTerminalTask"]}' the arguments would be:
            non_terminal_left=SomeNonTerminalTask, terminal_right=SomeTerminalTask, non_terminal_right=SomeOtherNonTerminalTask
            If the arguments do not match any rule in the grammar, the function returns "ERROR".""",
            parameters=types.Schema(
                type='OBJECT',
                properties={
                    'non_terminal_left': types.Schema(
                        type='string',
                        description='Non-terminal left-hand side symbol.',
                    ),
                    'terminal_right': types.Schema(
                        type='string',
                        description='Terminal right-hand side symbol.',
                    ),
                    'non_terminal_right': types.Schema(
                        type='string',
                        description='Non-terminal right-hand side symbol. Only required, if the rule contains a right-hand side non-terminal.',
                    )
                },
                required=['non_terminal_left', 'terminal_right'],
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
  
            # "tool_config": {"function_calling_config": {"mode": "any"}}
        }


    # remove_rule tool
    def remove_rule(self, non_terminal_left, terminal_right, non_terminal_right=None):
        # TODO remove symbols from terminals and non_terminals also, if they do not occur in rules anymore
        # TODO maybe use try catch and pass error messages to the model
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
        return
    
    def generate_next_response(self):
        
        response = self.client.models.generate_content(model=self.model, config=self.config, contents=self.contents)
        print("MODEL RESPONSE")
        print(response)
        self.contents.append(response.candidates[0].content)
        
        tool_call = response.candidates[0].content.parts[1].function_call
        
        if tool_call is not None:
            
            if tool_call.name == "remove_rule":
                result = self.remove_rule(**tool_call.args)
            if tool_call.name == "terminate":
                return False

            response_part = types.Part.from_function_response(
                name=tool_call.name,
                response={"result": result},
            )
        else:
            response_part = types.Part.from_text(text="No tool output")
            
        self.contents.append(types.Content(role="user", parts=[response_part]))
        return True
            
    
    def save_current_grammar(self, path):
        with open(path, "w") as f:
            json.dump(self.grammar, f, indent=4)
        print("Current grammar saved to", path)
    
    # TODO append to history after each response instead
    # TODO prettify
    def save_chat_history(self, path):
        print(self.contents)
        count_mess = 1
        with open(path, "a") as f:
            for content in self.contents:
                count_part = 1
                f.write("Message " + str(count_mess) + '\n')
                f.write("Role: " + str(content.role) + '\n')
                for part in content.parts:
                    f.write("Part " + str(count_part) + '\n')
                    f.write("Response text: " + str(part.text) + '\n')
                    f.write("Function call: " + str(part.function_call) + '\n')
                    f.write("Function response: " + str(part.function_response) + '\n\n')
                    count_part += 1
                f.write("-----------------------------------------------------\n\n")
                count_mess += 1
        print("Chat history saved to", path)
