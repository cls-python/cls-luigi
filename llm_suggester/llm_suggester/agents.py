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
        
        # TODO add task description to prompt

        self.grammar = grammar
        self.client = genai.Client(api_key="AIzaSyBzCMnDmfR9TLyvTBchKM6frGnHd3nxMHk")
        self.model = "gemini-2.0-flash"
        self.instructions = f"""
        The following is a regular tree grammar, which describes a set of all possible pipelines.
        \n{self.grammar}\n
        Your task is to iteratively chose which rule to eliminate, until the grammar can only produce one valid pipeline.
        To remove a rule you should use the remove_rule tool. After each removal the tool will return the updated grammar.
        Your goal is to remove as many rules, as necessary, to produce a grammar, that describes just a few (or even just one) meaningful and efficient pipelines for the regression task.
        This means, you should always think your decisions through and NOT guess! You should also always check, if an additional removal will be an improvement and if not stop on your own, by using the terminate tool.
        """
        
        # remove_rule tool
        def remove_rule(non_terminal1: str, terminal: str, non_terminal2: str) -> str: 
            """Removes the specified rule from the grammar and returns the updated grammar.
            To remove the rule '"SomeNonTerminalTask": {"SomeTerminalTask": ["SomeOtherNonTerminalTask"]}' the arguments would be:
            non_terminal1=SomeNonTerminalTask, terminal=SomeTerminalTask, non_terminal2=SomeOtherNonTerminalTask
            If the arguments do not match any rule in the grammar, the function returns "ERROR".
            
            Args:
                non_terminal1: non-terminal left-hand side symbol
                terminal: terminal right-hand side symbol
                non_terminal2: non-terminal right-hand side symbol, should be "" if the rule does not include a right-hand side non-terminal
            
            Returns:
                str: Updated grammar as a string, or "ERROR" if the rule could not be found.
            """
            
            # TODO remove symbols from terminals and non_terminals also, if they do not occur in rules anymore
            for rule in self.grammar["rules"]:
                if rule == non_terminal1:
                    if non_terminal2 is "":
                        self.grammar["rules"][rule].pop(terminal)
                        if self.grammar["rules"][rule] == {}: # if the terminal was the last for this rule, remove whole rule
                            self.grammar["rules"].pop(rule)
                    else:
                        self.grammar["rules"][rule][terminal].remove(non_terminal2) 
                    return str(self.grammar)
            return "ERROR"

        # terminate tool
        def terminate() -> None:
            """Terminates the chat and notifies the user, that a good grammar has been generated."""
            print("Termination requested. The grammar is now considered good enough.")
            return

        self.config = {
            "system_instruction": self.instructions,
            "tools": [remove_rule, terminate],
            # "automatic_function_calling": {"disable": True},
            "tool_config": {"function_calling_config": {"mode": "any"}}
        }

    def start_chat(self):
        print("Starting LLM chat...")
        chat = self.client.chats.create(model=self.model, config=self.config)   
        return chat
    
    def generate_next_response(self, chat, message):
        print("Generating response...")
        print("MESSAGE")
        print(message)
        return chat.send_message(message)
    
    def save_current_grammar(self, path):
        with open(path, "w") as f:
            json.dump(self.grammar, f, indent=4)
        print("Current grammar saved to", path)
    
    def save_chat_history(self, chat, path):
        count = 0   
        with open(path, "a") as f:
            for content in chat.get_history():
                part = content.parts[0]
                f.write("# " + str(count) + '\n')
                f.write("Role: " + str(content.role) + '\n')
                f.write("Response text: " + str(part.text) + '\n')
                f.write("Function call: " + str(part.function_call) + '\n')
                f.write("Function response: " + str(part.function_response) + '\n\n')
                count += 1
        print("Chat history saved to", path)

