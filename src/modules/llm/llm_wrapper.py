import os
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import time
import requests
from loguru import logger
import copy
import openai
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import asyncio
import aiohttp
import json

@dataclass
class LLMConfig:
    """Configuration for LLM parameters."""
    api_key: str
    # Default to an OpenAI-style model name; can be overridden by config/args
    model_name: str = "gpt-4.1"
    belief_dim: int = 64
    debug: bool = False
    max_retries: int = 6
    max_workers: int = 5
    # OpenAI-compatible base URL root (NOT the full endpoint).
    # Endpoints are derived as:
    # - {base_url}/chat/completions
    # - {base_url}/embeddings
    base_url: str = "https://pro.xiaoai.plus/v1"
    embeddings_base_url: Optional[str] = None
    # If your provider supports extra non-standard fields (e.g. repetition_penalty),
    # keep this enabled. If it errors, set False to send only OpenAI-standard params.
    allow_nonstandard_params: bool = True
    timeout: float = 60.0  # 增加默认超时时间
    retry_delay: float = 2.0  # 重试延迟
    request_delay: float = 0.1  # 请求间延迟

class DynamicParamNetwork(nn.Module):
    """
    Neural network for dynamically adjusting LLM parameters based on belief states.
    
    This network learns to generate temperature and top-p parameters for LLM sampling
    based on the current belief state of the system.
    """
    def __init__(self,
                 belief_dim: int,
                 t_min: float = 0.1,
                 t_max: float = 1.0,
                 p_min: float = 0.1,
                 p_max: float = 0.9):
        """
        Initialize the dynamic parameter network.
        
        Args:
            belief_dim: Dimension of belief state
            t_min: Minimum temperature value
            t_max: Maximum temperature value
            p_min: Minimum top-p value
            p_max: Maximum top-p value
        """
        super().__init__()
        self.belief_dim = belief_dim
        self.t_min = t_min
        self.t_max = t_max
        self.p_min = p_min
        self.p_max = p_max
        
        # Temperature generation network
        self.W_t = nn.Linear(belief_dim, 1)
        self.b_t = nn.Parameter(torch.zeros(1))
        
        # Top-p generation network
        self.W_p = nn.Linear(belief_dim, 1)
        self.b_p = nn.Parameter(torch.zeros(1))
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, belief_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate temperature and top-p parameters based on belief state.
        
        Args:
            belief_state: Current belief state
            
        Returns:
            Tuple of (temperature, top_p) parameters
        """
        # Generate temperature parameter
        t = self.sigmoid(self.W_t(belief_state) + self.b_t)
        temperature = self.t_min + (self.t_max - self.t_min) * t
        
        # Generate top-p parameter
        p = self.sigmoid(self.W_p(belief_state) + self.b_p)
        top_p = self.p_min + (self.p_max - self.p_min) * p
        
        return temperature.squeeze(), top_p.squeeze()

class APIHandler:
    """
    Handler for LLM API interactions with retry logic and response processing.
    """
    def __init__(self, config: LLMConfig):
        """
        Initialize the API handler.
        
        Args:
            config: LLM configuration
        """
        self.config = config
        # Default embeddings endpoint to the same provider unless overridden.
        if (not hasattr(self.config, "embeddings_base_url")) or (self.config.embeddings_base_url is None):
            self.config.embeddings_base_url = self.config.base_url
        self.sleep_times = [2 ** i for i in range(config.max_retries)]  # Exponential backoff

    def _chat_completions_url(self) -> str:
        return str(self.config.base_url).rstrip("/") + "/chat/completions"

    def _embeddings_url(self) -> str:
        return str(self.config.embeddings_base_url).rstrip("/") + "/embeddings"

    def _post_json(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=self.config.timeout)
        # Keep raw text for debugging
        try:
            data = resp.json()
        except Exception:
            data = {"_raw_text": resp.text, "_status_code": resp.status_code}
        # Raise for explicit HTTP errors (but still return parsed body for logging)
        if resp.status_code >= 400:
            raise requests.exceptions.HTTPError(f"HTTP {resp.status_code}", response=resp)
        return data
    
    def generate_with_references(
        self,
        model: str,
        messages: List[Dict],
        references: List[str] = [],
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        # OpenAI-compatible optional params
        response_format: Optional[Dict[str, Any]] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
    ) -> Optional[str]:
        """
        Generate response with reference integration.
        
        Args:
            model: Model identifier
            messages: List of message dictionaries
            references: List of reference strings
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            repetition_penalty: Repetition penalty
            
        Returns:
            Generated response or None if failed
        """
        if references:
            messages = self._inject_references(messages, references)
        
        return self.generate_together(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            response_format=response_format,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )
    
    # Backward-compatible name. This now sends an OpenAI-compatible Chat Completions request.
    def generate_together(
        self,
        model: str,
        messages: List[Dict],
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        streaming: bool = False,
        # OpenAI-compatible optional params
        response_format: Optional[Dict[str, Any]] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
    ) -> Optional[str]:
        """
        Generate response using Together API.
        
        Args:
            model: Model identifier
            messages: List of message dictionaries
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            repetition_penalty: Repetition penalty
            streaming: Whether to stream response
            
        Returns:
            Generated response or None if failed
        """
        output = None
        
        for sleep_time in self.sleep_times:
            try:
                if self.config.debug:
                    logger.debug(f"Sending messages ({len(messages)}) to {model}")
                
                payload: Dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": int(max_tokens),
                    "temperature": float(temperature) if float(temperature) > 1e-4 else 0.0,
                }
                
                if top_p is not None:
                    payload["top_p"] = float(top_p)
                if response_format is not None:
                    # OpenAI: {"type":"json_object"} etc.
                    payload["response_format"] = response_format
                if presence_penalty is not None:
                    payload["presence_penalty"] = float(presence_penalty)
                if frequency_penalty is not None:
                    payload["frequency_penalty"] = float(frequency_penalty)
                # Non-standard: keep for providers that accept it (Together-like)
                if repetition_penalty is not None and bool(getattr(self.config, "allow_nonstandard_params", True)):
                    payload["repetition_penalty"] = float(repetition_penalty)
                
                if streaming:
                    # Keep legacy streaming helper (may not work for all providers).
                    return self._stream_response(payload)

                url = self._chat_completions_url()
                # Some providers may not support OpenAI "response_format".
                # If it fails with a client error, retry once without response_format.
                try:
                    data = self._post_json(url, payload)
                except requests.exceptions.HTTPError as he:
                    status = None
                    body = ""
                    try:
                        if he.response is not None:
                            status = he.response.status_code
                            body = he.response.text
                    except Exception:
                        pass
                    if payload.get("response_format") is not None and status in (400, 422):
                        logger.warning(f"Provider rejected response_format, retrying without it. status={status} body={body[:200]}")
                        payload.pop("response_format", None)
                        data = self._post_json(url, payload)
                    else:
                        raise
                if isinstance(data, dict) and "error" in data:
                    logger.error(f"API Error: {data.get('error')}")
                    return None
                # OpenAI response: choices[0].message.content
                output = str(data["choices"][0]["message"]["content"]).strip()
                break
                
            except Exception as e:
                logger.error(f"API call failed: {str(e)}")
                if self.config.debug:
                    logger.debug(f"Messages: {messages}")
                logger.info(f"Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
        
        return output
    
    def generate_embeddings(self,
                            input_texts: Union[str, List[str]],
                            model: str) -> Optional[List[List[float]]]:
        """
        Generate embeddings using Together API.

        Args:
            input_texts: A single string or a list of strings to embed.
            model: The embedding model identifier.

        Returns:
            A list of embeddings (each embedding is a list of floats), or None if failed.
        """
        embeddings_list = None
        payload = {"input": input_texts, "model": model}

        for sleep_time in self.sleep_times:
            try:
                if self.config.debug:
                    logger.debug(f"Requesting embeddings for {len(input_texts) if isinstance(input_texts, list) else 1} text(s) from {model}")

                url = self._embeddings_url()
                response_data = self._post_json(url, payload)
                
                if "data" not in response_data or not isinstance(response_data["data"], list):
                    logger.error(f"API Error: 'data' field is missing or not a list in embeddings response: {response_data}")
                    return None

                embeddings_list = []
                for item in response_data["data"]:
                    if "embedding" in item and isinstance(item["embedding"], list):
                        embeddings_list.append(item["embedding"])
                    else:
                        logger.error(f"API Error: 'embedding' field missing or invalid in item: {item}")
                        # Continue to process other embeddings if possible, or decide to return None
                
                if not embeddings_list: # If all items failed or data was empty
                    logger.warning("No embeddings were successfully extracted.")
                    return None
                    
                break # Successful API call
                
            except requests.exceptions.HTTPError as http_err:
                # best-effort extract response body
                body = ""
                status = None
                try:
                    if http_err.response is not None:
                        status = http_err.response.status_code
                        body = http_err.response.text
                except Exception:
                    pass
                logger.error(f"HTTP error occurred during embedding generation: {http_err} - {body}")
                if status == 429: # Rate limit
                     logger.info(f"Rate limit hit. Retrying in {sleep_time}s...")
                     time.sleep(sleep_time)
                elif status is not None and status >= 500: # Server-side error
                    logger.info(f"Server error. Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                else: # Other client-side errors (4xx)
                    logger.error("Client-side error, not retrying for this attempt.")
                    # No break here, will go to next sleep_time or fail if it's the last retry
            except Exception as e:
                logger.error(f"API call for embeddings failed: {str(e)}")
                if self.config.debug:
                    logger.debug(f"Payload: {payload}")
                logger.info(f"Retrying in {sleep_time}s...")
                time.sleep(sleep_time) # General retry for other exceptions
        
        return embeddings_list
    
    def _stream_response(self, payload: Dict) -> Any:
        """
        Stream response from API.
        
        Args:
            payload: Request payload
            
        Returns:
            Streaming response
        """
        endpoint = "https://api.together.xyz/v1"
        client = openai.OpenAI(
            api_key=self.config.api_key,
            base_url=endpoint
        )
        return client.chat.completions.create(**payload, stream=True)
    
    def _inject_references(self, messages: List[Dict], references: List[str]) -> List[Dict]:
        """
        Inject reference information into messages.
        
        Args:
            messages: Original messages
            references: Reference strings to inject
            
        Returns:
            Messages with injected references
        """
        messages = copy.deepcopy(messages)
        system_prompt = """You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from models:"""
        
        for i, ref in enumerate(references, 1):
            system_prompt += f"\n{i}. {ref}"
        
        if messages[0]["role"] == "system":
            messages[0]["content"] += "\n\n" + system_prompt
        else:
            messages.insert(0, {"role": "system", "content": system_prompt})
        
        return messages

class ImprovedLLMWrapper:
    def __init__(self,
                 api_key: str,
                 model_name: str = "gpt-4.1",
                 belief_dim: int = 64,
                 encoding_dim: int = 384,
                 debug: bool = False,
                 t_min: float = 0.1,
                 t_max: float = 2.0,
                 rp_min: float = 1.0,
                 rp_max: float = 1.5,
                 timeout: float = 60.0,
                 base_url: Optional[str] = None,
                 embeddings_base_url: Optional[str] = None,
                 allow_nonstandard_params: bool = True):
        """
      
        """
        config = LLMConfig(
            api_key=api_key,
            model_name=model_name,
            belief_dim=belief_dim,
            debug=debug,
            timeout=timeout,
            retry_delay=2.0,
            base_url=(base_url or LLMConfig.base_url),
            embeddings_base_url=embeddings_base_url,
            allow_nonstandard_params=bool(allow_nonstandard_params),
        )
        
        self.model_name = model_name
        self.api_handler = APIHandler(config)
        self.param_network = DynamicParamNetwork(
            belief_dim=belief_dim,
            t_min=t_min,
            t_max=t_max,
            p_min=rp_min,
            p_max=rp_max
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.param_network.to(self.device)
        
        
        self.encoding_dim = encoding_dim
        
        # 缓存机制
        self.response_cache = {}
        self.max_cache_size = 1000
        
        # 统计信息
        self.request_count = 0
        self.timeout_count = 0
        self.success_count = 0
        
        logger.info(f"ImprovedLLMWrapper initialized with model: {model_name}, timeout: {timeout}s")
    
    def encode_text(self, text: str) -> torch.Tensor:
        """
 
        
        Args:
            text
            
        Returns:
            Encoded tensor
        """
        try:
            
            words = text.split()
            n_words = len(words)
            
           
            features = []
            
        
            word_lengths = [len(w) for w in words]
            avg_word_len = sum(word_lengths) / (n_words + 1e-6)
            max_word_len = max(word_lengths) if word_lengths else 0
            
         
            char_counts = {}
            for c in text.lower():
                if c.isalnum():
                    char_counts[c] = char_counts.get(c, 0) + 1
            
            
            num_count = sum(1 for c in text if c.isdigit())
            
           
            encoding = torch.zeros(self.encoding_dim)
            
            
            basic_features = [
                len(text),           
                n_words,             
                avg_word_len,       
                max_word_len,        
                num_count,           
                len(char_counts),    
                sum(char_counts.values()),  
                len([w for w in words if w[0].isupper()]),  
            ]
            
            encoding[:len(basic_features)] = torch.tensor(basic_features)
            
            
            char_idx = len(basic_features)
            for i, (char, count) in enumerate(sorted(char_counts.items())):
                if char_idx + i < self.encoding_dim:
                    encoding[char_idx + i] = count / len(text)
            
            
            pos_idx = char_idx + len(char_counts)
            for i, word in enumerate(words):
                if pos_idx + i < self.encoding_dim:
                    encoding[pos_idx + i] = i / n_words
            
           
            encoding = (encoding - encoding.mean()) / (encoding.std() + 1e-6)
            
            return encoding.to(self.device)
            
        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            return torch.zeros(self.encoding_dim).to(self.device)
            
    def _make_request_with_retry(self, payload: Dict[str, Any]) -> Optional[str]:
        """
        带重试机制的请求方法
        """
        headers = {
            "Authorization": f"Bearer {self.api_handler.config.api_key}",
            "Content-Type": "application/json"
        }
        
        url = f"{self.api_handler.config.base_url}/chat/completions"
        
        for attempt in range(self.api_handler.config.max_retries):
            try:
                self.request_count += 1
                
                if self.api_handler.config.debug:
                    logger.debug(f"LLM API请求 (尝试 {attempt + 1}/{self.api_handler.config.max_retries}): {payload['messages'][0]['content'][:50]}...")
                
                start_time = time.time()
                
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.api_handler.config.timeout
                )
                
                end_time = time.time()
                request_time = end_time - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    content = result['choices'][0]['message']['content']
                    self.success_count += 1
                    
                    if self.api_handler.config.debug:
                        logger.debug(f"LLM API成功 (用时: {request_time:.2f}s): {content[:50]}...")
                    
                    return content
                else:
                    logger.warning(f"LLM API返回错误状态 {response.status_code}: {response.text}")
                    if attempt == self.api_handler.config.max_retries - 1:
                        return None
                    
            except requests.exceptions.Timeout:
                self.timeout_count += 1
                logger.warning(f"LLM API超时 (尝试 {attempt + 1}/{self.api_handler.config.max_retries})")
                if attempt == self.api_handler.config.max_retries - 1:
                    logger.error("LLM API多次超时，返回默认响应")
                    return "Unable to generate response due to timeout."
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"LLM API请求异常 (尝试 {attempt + 1}/{self.api_handler.config.max_retries}): {e}")
                if attempt == self.api_handler.config.max_retries - 1:
                    return None
                    
            except Exception as e:
                logger.error(f"LLM API未知错误: {e}")
                if attempt == self.api_handler.config.max_retries - 1:
                    return None
            
            # 重试前等待
            if attempt < self.api_handler.config.max_retries - 1:
                time.sleep(self.api_handler.config.retry_delay * (attempt + 1))  # 指数退避
        
        return None
    
    def generate_response(self, 
                         prompt: str,
                         strategy: Optional[str] = None,
                         belief_state: Optional[torch.Tensor] = None,
                         max_tokens: int = 2048,
                         references: List[str] = [],
                         temperature: Optional[float] = None,
                         repetition_penalty: Optional[float] = None,
                         top_p: Optional[float] = None,
                         llm_model_name: Optional[str] = None,
                         # OpenAI-compatible: request strict JSON output when supported by provider
                         response_format: Optional[Dict[str, Any]] = None) -> str:
       
        final_temp = temperature
        final_rp = repetition_penalty
        final_top_p = top_p

        messages = []
        if strategy:
            messages.append({"role": "system", "content": f"Strategy: {strategy}"})

        messages.append({"role": "user", "content": prompt})

        if final_temp is None or final_rp is None:
            if belief_state is not None and self.param_network is not None:
                generated_temp, generated_rp_or_top_p = self.param_network(belief_state)
                
                if final_temp is None:
                    final_temp = generated_temp.item()
                
                if final_rp is None:
                    final_rp = generated_rp_or_top_p.item()

        if final_temp is None:
            final_temp = 0.7
        if final_rp is None:
            final_rp = 1.0

        actual_model_name = llm_model_name if llm_model_name is not None else self.model_name

        response_content = self.api_handler.generate_with_references(
            model=actual_model_name,
            messages=messages,
            references=references,
            max_tokens=max_tokens,
            temperature=final_temp,
            top_p=final_top_p,
            repetition_penalty=final_rp,
            response_format=response_format,
        )
        
        return response_content if response_content is not None else "Error: Could not generate response."
        

    def update_param_network(self,
                           belief_states: torch.Tensor,
                           rewards: torch.Tensor,
                           optimizer: torch.optim.Optimizer) -> float:
        """
        Update parameter network based on rewards.
        
        Args:
            belief_states: Batch of belief states
            rewards: Batch of rewards
            optimizer: Optimizer instance
            
        Returns:
            Loss value
        """
        temperature, top_p = self.param_network(belief_states)
        
        # Compute loss based on rewards
        loss = -torch.mean(rewards * (temperature + top_p))
        
        # Update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def batch_generate(self,
                      prompts: List[str],
                      belief_states: torch.Tensor,
                      references: Optional[List[List[str]]] = None,
                      max_tokens: int = 2048) -> List[str]:
        """
        Generate responses in batch.
        
        Args:
            prompts: List of input prompts
            belief_states: Batch of belief states
            references: Optional references for each prompt
            max_tokens: Maximum tokens per response
            
        Returns:
            List of generated responses
        """
        if references is None:
            references = [[] for _ in prompts]
        
        with torch.no_grad():
            temperatures, top_ps = self.param_network(belief_states)
        
        responses = []
        with ThreadPoolExecutor(max_workers=min(len(prompts), 5)) as executor:
            futures = []
            for prompt, refs, temp, p in zip(prompts, references, temperatures, top_ps):
                future = executor.submit(
                    self.api_handler.generate_with_references,
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    references=refs,
                    max_tokens=max_tokens,
                    temperature=temp.item(),
                    top_p=p.item()
                )
                futures.append(future)
            
            for future in futures:
                try:
                    response = future.result()
                    responses.append(response)
                except Exception as e:
                    logger.error(f"Error in batch generation: {str(e)}")
                    responses.append(None)
        
        return responses