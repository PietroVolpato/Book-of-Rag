# src/galoispy/utils/openai_config.py
import json
from langchain_openai import AzureChatOpenAI
from typing import Dict, Any

def _openai_azure_model_config(api_version: str, deployment: str, endpoint: str, api_key: str, temperature: float, max_tokens: int, top_p: float, frequency_penalty: float, presence_penalty: float) -> AzureChatOpenAI:
    """
    OpenAI model configuration for Azure with customizable parameters:
    - `api_version`: The API version to use for the Azure OpenAI model.
    - `deployment`: The deployment name for the Azure OpenAI model.
    - `endpoint`: The endpoint URL for the Azure OpenAI model.
    - `api_key`: The API key for accessing the Azure OpenAI model.
    - `temperature`: The temperature setting for the Azure OpenAI model.
    - `max_tokens`: The maximum number of tokens for the Azure OpenAI model.
    - `top_p`: The top-p setting for the Azure OpenAI model.
    - `frequency_penalty`: The frequency penalty setting for the Azure OpenAI model.
    - `presence_penalty`: The presence penalty setting for the Azure OpenAI model.
    
    :param api_version: The API version to use.
    :type api_version: str
    :param deployment: The deployment name.
    :type deployment: str
    :param endpoint: The endpoint URL.
    :type endpoint: str
    :param api_key: The API key.
    :type api_key: str
    :param temperature: The temperature setting.
    :type temperature: float
    :param max_tokens: The maximum number of tokens.
    :type max_tokens: int
    :param top_p: The top-p setting.
    :type top_p: float
    :param frequency_penalty: The frequency penalty setting.
    :type frequency_penalty: float
    :param presence_penalty: The presence penalty setting.
    :type presence_penalty: float
    :return: The configured AzureChatOpenAI model.
    :rtype: AzureChatOpenAI
    """
    # Model options
    options: Dict[str, Any] = {
        'temperature': temperature,
        'max_tokens': max_tokens,
        'top_p': top_p,
        'frequency_penalty': frequency_penalty,
        'presence_penalty': presence_penalty
    }

    openai_model = AzureChatOpenAI(
        api_version=api_version,
        azure_deployment=deployment,
        azure_endpoint=endpoint,
        api_key=api_key,
        **options
    )

    # Return the model
    return openai_model

def openai_azure_model_init(api_key:str, endpoint:str, deployment:str, api_version:str, config_schema:str) -> AzureChatOpenAI:
    """
    Initialize the OpenAI model from a configuration schema in JSON.\n
    It extracts:
    - the model name.
    - the model temperature.
    - the maximum number of tokens.
    - the top-p setting.
    
    It requires also:
    - the API key.
    - the endpoint URL.
    - the deployment name.
    
    :param api_key: The API key.
    :type api_key: str
    :param endpoint: The endpoint URL.
    :type endpoint: str
    :param deployment: The deployment name.
    :type deployment: str
    :param api_version: The API version.
    :type api_version: str
    :param config_schema: The configuration schema in JSON.
    :type config_schema: str
    :return: The initialized AzureChatOpenAI model.
    :rtype: AzureChatOpenAI
    """
    schema = json.loads(config_schema)
    if "temperature" in schema:
        temperature = float(schema["temperature"])
    else:
        temperature = 0.0
    if "max_tokens" in schema:
        max_tokens = int(schema["max_tokens"])
    else:
        max_tokens = 1024
    if "top_p" in schema:
        top_p = float(schema["top_p"])
    else:
        top_p = 1.0
    if "frequency_penalty" in schema:
        frequency_penalty = float(schema["frequency_penalty"])
    else:
        frequency_penalty = 0.0
    if "presence_penalty" in schema:
        presence_penalty = float(schema["presence_penalty"])
    else:
        presence_penalty = 0.0
    
    # Initialize the model
    openai_model = _openai_azure_model_config(api_version=api_version, deployment=deployment, endpoint=endpoint, api_key=api_key, temperature=temperature, max_tokens=max_tokens, top_p=top_p, frequency_penalty=frequency_penalty, presence_penalty=presence_penalty)

    # Return the model
    return openai_model