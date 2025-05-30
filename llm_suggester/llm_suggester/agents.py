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
    
    def __init__(self, grammar):

        self.grammar = grammar
        self.client = genai.Client(api_key="AIzaSyBzCMnDmfR9TLyvTBchKM6frGnHd3nxMHk")
        self.model = "gemini-2.0-flash"
        self.instructions = f"""
        The following is a regular tree grammar, which describes a set of all possible pipelines.
        \n{self.grammar}\n
        Your task is to iteratively chose which rule to eliminate, until the grammar can only produce one valid pipeline.
        To remove a rule you should use the remove_rule tool. After each removal the tool will return the number of pipelines that can still be produced by the grammar.
        When you get to 1, the final version of the grammar should produce an efficient and meaningful pipeline for the regression task.
        """
        def remove_rule(non_terminal1: str, terminal: str, non_terminal2: str) -> int: 
            """Removes the specified rule from the grammar and returns the number of pipelines, that can still be produced from the updated grammar.
            To remove the rule '"SomeNonTerminalTask": {"SomeTerminalTask": ["SomeOtherNonTerminalTask"]}' the arguments would be:
            non_terminal1=SomeNonTerminalTask, terminal=SomeTerminalTask, non_terminal2=SomeOtherNonTerminalTask
            If the arguments do not match any rule in the grammar, the function returns -1.
            
            Args:
                non_terminal1: non-terminal left-hand side symbol
                terminal: terminal right-hand side symbol
                non_terminal2: non-terminal right-hand side symbol, should be None if the rule does not include a right-hand side non-terminal
            """
            
            for rule in self.grammar["rules"]:
                if rule == non_terminal1:
                    if non_terminal2 is None:
                        self.grammar["rules"][rule].pop(terminal)
                        if self.grammar["rules"][rule] == {}: # if the terminal was the last for this rule, remove whole rule
                            self.grammar["rules"].pop(rule)
                    else:
                        self.grammar["rules"][rule][terminal].remove(non_terminal2) 
                    num_pipelines = 3 # TODO should probably use cls to determine this
                    return num_pipelines
            return -1

        self.config = types.GenerateContentConfig(
            system_instruction=self.instructions,
            tools=[remove_rule])

    def start_chat(self):
        print("Starting LLM chat...")
        chat = self.client.chats.create(model=self.model, config=self.config)   
        return chat
    
    def save_chat_history(self, chat, path):
        count = 0
        with open(path, "a") as f:
            for cont in chat.get_history():
                part = cont.parts[0]
                f.write("# " + str(count) + '\n')
                f.write("Response text: " + str(part.text) + '\n')
                f.write("Function call: " + str(part.function_call) + '\n')
                f.write("Function response: " + str(part.function_response) + '\n\n')
                count += 1
        print("Chat history saved to", path)

