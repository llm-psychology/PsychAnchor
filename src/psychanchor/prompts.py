from pathlib import Path
from personadough.runtime.prompts import PromptRegistry

prompt_loader = PromptRegistry(Path(__file__).with_name("prompts.json"))
get_prompt = prompt_loader.get_prompt
get_system_prompt = prompt_loader.get_system_prompt
get_user_prompt = prompt_loader.get_user_prompt
