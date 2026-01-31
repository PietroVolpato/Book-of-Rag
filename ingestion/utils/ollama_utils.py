# src/galoispy/utils/ollama_utils.py
import json
import ollama
import os
import platform
import psutil
import re
import subprocess
import time
from typing import Any, Dict, Tuple
# Langchain packages
from langchain_ollama import ChatOllama

def is_ollama_active() -> bool:
    """
    Check if the Ollama server is active.\n
    Returns True if the service is running, False otherwise.
    
    :return: True if Ollama is active, False otherwise.
    :rtype: bool
    """
    list = psutil.pids()
    for i in range(0, len(list)):
        try:
            p = psutil.Process(list[i])
            if (platform.system() == "Windows" and p.cmdline()[0].find("ollama.exe") != -1) or (platform.system() != "Windows" and "ollama" in p.cmdline()[0]):
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, IndexError):
            continue
    return False

def start_ollama() -> None:
    """
    Start the Ollama server if it is not already running.
    """
    if not is_ollama_active():
        # Start the Ollama server
        if platform.system() == "Windows":
            os.system("start /B ollama serve > NUL 2>&1")
        else:
            os.system("ollama serve & > /dev/null 2>&1 &")
        # Wait for a few seconds to ensure the service starts properly
        time.sleep(2)

def stop_ollama() -> None:
    """
    Stop the Ollama server if it is running.
    """
    if is_ollama_active():
        # Stop the Ollama server
        for proc in psutil.process_iter(['name', 'cmdline']):
            try:
                if platform.system() == "Windows":
                    exe = os.path.basename(proc.info.get('exe', '')).lower()
                    name = proc.info.get('name')
                    if name == 'ollama.exe' or exe == 'ollama.exe':
                        proc.kill()
                        time.sleep(1)
                else:
                    if 'ollama' in proc.info.get('name', '') or any('ollama' in cmd for cmd in proc.info.get('cmdline', [])):
                        proc.kill()
                        time.sleep(1)
            except (psutil.NoSuchProcess, psutil.AccessDenied, IndexError):
                continue
            except Exception as e:
                raise RuntimeError(f"An error occurred while checking the Ollama process.") from e

def _get_models_information(models: ollama._types.ListResponse) -> list[Tuple[str, str, str, str]]:
    """
    Get information about the available models.

    Example of a model information:
    ```
    Model(
        model='deepseek-r1:1.5b', 
        modified_at=datetime.datetime(2025, 8, 5, 13, 34, 49, 479889, tzinfo=TzInfo(+02:00)), 
        digest='e0979632db5a88d1a53884cb2a941772d10ff5d055aabaa6801c4e36f3a6c2d7', 
        size=1117322768, 
        details=ModelDetails(
            parent_model='', 
            format='gguf', 
            family='qwen2', 
            families=['qwen2'], 
            parameter_size='1.8B', 
            quantization_level='Q4_K_M'
        )
    )
    ```
    
    :param models: The list of models from Ollama.
    :type models: ollama._types.ListResponse
    :return: A list of tuples containing model information (name, size, family, parameter size).
    :rtype: list[Tuple[str, str, str, str]]
    """
    # Unpack the model details
    for model in models:
        (left, *model_details) = model if isinstance(model, tuple) else (model, {})
    model_list = str(model_details).split("), Model(")
    models_info = []
    # Extract model details using regex
    for m in model_list:
        name = re.search(r"model='([^']+)'", str(m))
        size = re.search(r"size=(\d+)", str(m))
        family = re.search(r"family='([^']+)'", str(m))
        param = re.search(r"parameter_size='([^']+)'", str(m))
        # Save the model details in a tuple
        models_info.append((name.group(1), size.group(1), family.group(1), param.group(1)))
    return models_info

def list_models_with_informations() -> list[Tuple[str, str]]:
    """
    List of downloaded models from Ollama with their information.
    
    :return: A set of tuples containing model name and its formatted information.
    :rtype: list[Tuple[str, str]]
    """
    downloaded_models = set()
    available_models = ollama.list()
    models_info = _get_models_information(available_models)
    for model in models_info:
        model_name = model[0] if model[0] else 'N/A'
        model_size = float("{:.2f}".format(int(model[1])/1024/1024/1024)) if model[1] else 'N/A'
        model_family = model[2] if model[2] else 'N/A'
        model_param = model[3] if model[3] else 'N/A' 
        # Save the model name to the set of downloaded models
        downloaded_models.add((model_name ,f"{model_name:<15}\t[size: {model_size} GB, family: {model_family}, param: {model_param}]"))
    return downloaded_models

def list_models() -> set:
    """
    List of downloaded models from Ollama.
    
    :return: A set of downloaded model names.
    :rtype: set
    """
    downloaded_models = set()
    while True:
        try:
            available_models = ollama.list()
            models_info = _get_models_information(available_models)
            for model in models_info:
                model_name = model[0] if model[0] else 'N/A'
                # Save the model name to the set of downloaded models
                downloaded_models.add(model_name)
            break
        except ConnectionError as e:
            # Stop and restart the server
            stop_ollama()
            time.sleep(2)
            start_ollama()
        except Exception as ex:
            raise RuntimeError(f"An unexpected error occurred while listing models.") from ex
    # Return the set of downloaded models
    return downloaded_models

def _model_selection(model_name: str, download: bool) -> str:
    """
    Select a model from the list of downloaded models.\n
    If it is not present, it will download it.
    
    :param model_name: The name of the model to select.
    :type model_name: str
    :param download: Whether to download the model if not present.
    :type download: bool
    :return: The selected model name.
    :rtype: str
    """
    downloaded_models = list_models()
    if model_name not in downloaded_models:
        if download:
            # Download the model
            result = subprocess.run(
                ["ollama", "pull", model_name],
                # Capture the command output in `result.stdout` or `result.stderr`
                capture_output=True,
                # Decode the output as a string
                text=True,
                # UTF-8 encoding
                encoding="utf-8",
                # Ignore the character errors
                errors="ignore"
            )
            if result.returncode != 0:
                stop_ollama()
                raise RuntimeError(f"I can't find the model {model_name}. Please verify the correctness of the model name.")
            else:
                return model_name
        else:
            raise RuntimeError(f"The model {model_name} is not downloaded.")
    return model_name

def _ollama_model_config(model_name: str, temperature: float, think: bool, top_p: float, frequency_penalty: float, presence_penalty: float) -> ChatOllama:
    """
    Configure a specific model for Ollama with some parameters.
    - `model_name`: the model name.
    - `temperature`: the model temperature [0.0 : 1.0].
    - `think`: whether to enable "thinking" mode [True, False].
    - `top_p`: the top-p sampling parameter.
    - `frequency_penalty`: the frequency penalty parameter.
    - `presence_penalty`: the presence penalty parameter.
    
    :param model_name: The name of the model to configure.
    :type model_name: str
    :param temperature: The model temperature.
    :type temperature: float
    :param think: Whether to enable "thinking" mode.
    :type think: bool
    :param top_p: The top-p sampling parameter.
    :type top_p: float
    :param frequency_penalty: The frequency penalty parameter.
    :type frequency_penalty: float
    :param presence_penalty: The presence penalty parameter.
    :type presence_penalty: float
    :return: The configured ChatOllama model.
    :rtype: ChatOllama
    """
    # Model options
    options: Dict[str, Any] = {
        # Model temperature [0.0 : 1.0]
        'temperature': temperature,
        # Top-p sampling parameter
        'top_p': top_p,
        # Frequency penalty parameter
        'frequency_penalty': frequency_penalty,
        # Presence penalty parameter
        'presence_penalty': presence_penalty
    }
    if think:
        try:
            # Try to create the model with "thinking" mode enabled
            options['think'] = True
            ollama_model = ChatOllama(model=model_name, **options)
        except Exception:
            # If the model does not support "thinking" mode, disable it
            del options['think']
            ollama_model = ChatOllama(model=model_name, **options)
    else:
        ollama_model = ChatOllama(model=model_name, **options)

    # Return the model
    return ollama_model

def ollama_model_init(config_schema:json) -> ChatOllama:
    """
    Initialize the Ollama model from a configuration schema in JSON.\n
    It extracts:
    - The model name.
    - If the model has to be downloaded or not.
    - The model temperature.
    - If the "thinking" mode has to be enabled or not.
    - The top-p setting.
    - The frequency penalty setting.
    - The presence penalty setting.
    
    :param config_schema: The configuration schema in JSON.
    :type config_schema: json
    :return: The initialized ChatOllama model.
    :rtype: ChatOllama
    """
    # Load the configuration schema
    schema = json.loads(config_schema)
    if "model" in schema:
        model_name = schema["model"]
    else:
        raise RuntimeError("The configuration schema must contain the 'model' field.")
    if "download" in schema:
        download = schema["download"]
    else:
        download = False
    if "temperature" in schema:
        temperature = float(schema["temperature"])
    else:
        temperature = 0.0
    if "think" in schema:
        think = bool(schema["think"])
    else:
        think = False
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

    # Start the Ollama server if it is not already running
    start_ollama()
    
    # Select the model
    model_name = _model_selection(model_name=model_name, download=download)
    
    # Configure the model
    ollama_model = _ollama_model_config(model_name=model_name, temperature=temperature, think=think, top_p=top_p, frequency_penalty=frequency_penalty, presence_penalty=presence_penalty)
    
    # Return the model
    return ollama_model