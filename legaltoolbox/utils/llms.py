import omegaconf
import numpy as np
import hydra
from datetime import datetime
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import logging

logger = logging.getLogger(__name__)

def format_prompt(prompt: str | list | dict | omegaconf.dictconfig.DictConfig):
    if type(prompt) == str:
        if prompt.startswith("{'system':") and prompt.endswith("}"):
            prompt = eval(prompt)
        elif prompt.startswith("[('system',") and prompt.endswith(")]"):
            prompt = eval(prompt)
            
    if isinstance(prompt, dict) or isinstance(prompt, omegaconf.dictconfig.DictConfig):
        prompt = [(k,v) for k,v in prompt.items()]

    if type(prompt) == str:
        prompt = ChatPromptTemplate.from_template(prompt)
    elif type(prompt) == list:
        prompt = ChatPromptTemplate.from_messages(prompt)
    else:
        raise ValueError("Prompt must be a string or a list of messages")
    
    logger.info(f"Initializing chain with prompt: {prompt}")
    return prompt

def get_chain(prompt: str | list | dict | omegaconf.dictconfig.DictConfig, llm: dict | omegaconf.dictconfig.DictConfig, run_name: str, max_concurrency : int = 20, num_ctx_input=None):    
    logger.info(f"Initializing chain with llm: {llm}")

    if num_ctx_input is not None and "num_ctx" in llm.keys() and llm["num_ctx"] is None:
        llm["num_ctx"] = 2**(int(np.log2(2 * num_ctx_input)) + 1)
        logger.info(f"Running with a {llm['num_ctx']} context window due to input (2 x input tokens)")

    return (
        format_prompt(prompt)
        | hydra.utils.instantiate(llm)
        | StrOutputParser()
    ).with_config({"run_name": f"POCDGT - {run_name} - {datetime.now().isoformat()}", "max_concurrency":max_concurrency})