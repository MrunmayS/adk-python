from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Any
from typing import AsyncGenerator
from typing import cast
from typing import Dict
from typing import Generator
from typing import Iterable
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Union

from google.genai import types
from pydantic import BaseModel
from pydantic import Field
import requests
from typing_extensions import override

from .base_llm import BaseLlm
from .llm_request import LlmRequest
from .llm_response import LlmResponse

logger = logging.getLogger("google_adk." + __name__)

_NEW_LINE = "\n"
_EXCLUDED_PART_FIELD = {"inline_data": {"data"}}
_DEFAULT_BASE_URL = "https://api.us-east-1.langdb.ai"


class FunctionChunk(BaseModel):
  id: Optional[str]
  name: Optional[str]
  args: Optional[str]
  index: Optional[int] = 0


class TextChunk(BaseModel):
  text: str


class UsageMetadataChunk(BaseModel):
  prompt_tokens: int
  completion_tokens: int
  total_tokens: int


class LangDBClient:
  """Client for making HTTP requests to LangDB API."""

  def __init__(self, api_key: str, base_url: str = _DEFAULT_BASE_URL, extra_headers: Optional[Dict[str, str]] = None):
    """Initialize LangDB client.
    
    Args:
      api_key: The API key for authentication.
      base_url: The base URL for the LangDB API.
      extra_headers: Additional headers to include in requests.
    """
    self.api_key = api_key
    self.base_url = base_url.rstrip('/')
    self.extra_headers = extra_headers or {}

  def _get_headers(self) -> Dict[str, str]:
    """Get headers for API requests."""
    headers = {
      "Authorization": f"Bearer {self.api_key}",
      "Content-Type": "application/json",
    }
    headers.update(self.extra_headers)
    return headers

  async def chat_completion(self, payload: Dict[str, Any], stream: bool = False) -> Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]:
    """Make a chat completion request to LangDB.
    
    Args:
      payload: The request payload.
      stream: Whether to stream the response.
      
    Returns:
      The response from the API.
    """
    url = f"{self.base_url}/v1/chat/completions"
    headers = self._get_headers()
    payload["stream"] = stream
    
    # Use asyncio to run the blocking request in a thread pool
    if stream:
      return await asyncio.get_event_loop().run_in_executor(
        None, self._stream_request, url, headers, payload
      )
    else:
      return await asyncio.get_event_loop().run_in_executor(
        None, self._blocking_request, url, headers, payload
      )

  def _blocking_request(self, url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
    """Make a blocking HTTP request."""
    logger.debug(f"Request URL: {url}")
    logger.debug(f"Request Headers: {headers}")
    logger.debug(f"Request Payload: {json.dumps(payload, indent=2)}")
    
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    
    logger.debug(f"Response Status: {response.status_code}")
    logger.debug(f"Response Headers: {dict(response.headers)}")
    
    if not response.ok:
      try:
        error_body = response.json()
        logger.error(f"Error Response: {json.dumps(error_body, indent=2)}")
      except:
        logger.error(f"Error Response Text: {response.text}")
    
    response.raise_for_status()
    return response.json()

  def _stream_request(self, url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
    """Make a streaming HTTP request."""
    response = requests.post(url, headers=headers, json=payload, stream=True, timeout=60)
    response.raise_for_status()
    
    for line in response.iter_lines():
      if line:
        line_str = line.decode('utf-8')
        if line_str.startswith('data: '):
          data_str = line_str[6:]
          if data_str.strip() == '[DONE]':
            break
          try:
            yield json.loads(data_str)
          except json.JSONDecodeError:
            logger.warning(f"Failed to parse streaming response: {data_str}")


def _safe_json_serialize(obj) -> str:
  """Convert any Python object to a JSON-serializable type or string.
  
  Args:
    obj: The object to serialize.
    
  Returns:
    The JSON-serialized object string or string.
  """
  try:
    return json.dumps(obj, ensure_ascii=False)
  except (TypeError, OverflowError):
    return str(obj)


def _content_to_langdb_message(content: types.Content) -> Dict[str, Any]:
  """Convert a types.Content to a LangDB message format.
  
  Args:
    content: The content to convert.
    
  Returns:
    A LangDB message dictionary.
  """
  role = "assistant" if content.role in ["model", "assistant"] else "user"
  
  # Handle tool/function responses
  if any(part.function_response for part in content.parts):
    tool_messages = []
    for part in content.parts:
      if part.function_response:
        tool_messages.append({
          "role": "tool",
          "tool_call_id": part.function_response.id,
          "content": _safe_json_serialize(part.function_response.response)
        })
    return tool_messages if len(tool_messages) > 1 else tool_messages[0]
  
  # Handle regular messages
  message_content = []
  tool_calls = []
  
  for part in content.parts:
    if part.text:
      message_content.append({
        "type": "text",
        "text": part.text
      })
    elif part.inline_data and part.inline_data.data and part.inline_data.mime_type:
      base64_string = base64.b64encode(part.inline_data.data).decode("utf-8")
      if part.inline_data.mime_type.startswith("image"):
        message_content.append({
          "type": "image_url",
          "image_url": {
            "url": f"data:{part.inline_data.mime_type};base64,{base64_string}"
          }
        })
      else:
        raise ValueError(f"Unsupported media type: {part.inline_data.mime_type}")
    elif part.function_call:
      tool_calls.append({
        "id": part.function_call.id or "",
        "type": "function",
        "function": {
          "name": part.function_call.name,
          "arguments": _safe_json_serialize(part.function_call.args)
        }
      })
  
  # Build the message
  message = {"role": role}
  
  if len(message_content) == 1 and message_content[0]["type"] == "text":
    message["content"] = message_content[0]["text"]
  elif message_content:
    message["content"] = message_content
  
  if tool_calls:
    message["tool_calls"] = tool_calls
    
  return message


def _normalize_schema_types(schema_dict: Dict[str, Any]) -> Dict[str, Any]:
  """Recursively normalize type fields to lowercase for JSON Schema compatibility.
  
  Args:
    schema_dict: The schema dictionary to normalize.
    
  Returns:
    The normalized schema dictionary.
  """
  if "type" in schema_dict:
    schema_dict["type"] = schema_dict["type"].lower()
  
  # Handle nested properties
  if "properties" in schema_dict:
    for prop_key, prop_value in schema_dict["properties"].items():
      if isinstance(prop_value, dict):
        schema_dict["properties"][prop_key] = _normalize_schema_types(prop_value)
  
  # Handle array items
  if "items" in schema_dict and isinstance(schema_dict["items"], dict):
    schema_dict["items"] = _normalize_schema_types(schema_dict["items"])
  
  # Handle anyOf (union types)
  if "any_of" in schema_dict and isinstance(schema_dict["any_of"], list):
    schema_dict["any_of"] = [
      _normalize_schema_types(item) if isinstance(item, dict) else item
      for item in schema_dict["any_of"]
    ]
  
  return schema_dict


def _function_declaration_to_langdb_tool(func_decl: types.FunctionDeclaration) -> Dict[str, Any]:
  """Convert a types.FunctionDeclaration to LangDB tool format.
  
  Args:
    func_decl: The function declaration to convert.
    
  Returns:
    A LangDB tool dictionary.
  """
  tool = {
    "type": "function",
    "function": {
      "name": func_decl.name,
      "description": func_decl.description or "",
    }
  }
  
  if func_decl.parameters and func_decl.parameters.properties:
    properties = {}
    for key, value in func_decl.parameters.properties.items():
      prop = value.model_dump(exclude_none=True)
      # Recursively normalize types to ensure JSON Schema compatibility
      prop = _normalize_schema_types(prop)
      properties[key] = prop
    
    tool["function"]["parameters"] = {
      "type": "object",
      "properties": properties
    }
    
    if func_decl.parameters.required:
      tool["function"]["parameters"]["required"] = func_decl.parameters.required
  
  return tool


def _langdb_response_to_chunk(response: Dict[str, Any]) -> Generator[Tuple[Optional[Union[TextChunk, FunctionChunk, UsageMetadataChunk]], Optional[str]], None, None]:
  """Convert a LangDB response to chunks.
  
  Args:
    response: The LangDB response.
    
  Yields:
    Tuples of chunks and finish reasons.
  """
  if "choices" not in response or not response["choices"]:
    yield None, None
    return
    
  choice = response["choices"][0]
  finish_reason = choice.get("finish_reason")
  
  # Handle streaming delta
  if "delta" in choice:
    delta = choice["delta"]
    
    if "content" in delta and delta["content"]:
      yield TextChunk(text=delta["content"]), finish_reason
    
    if "tool_calls" in delta and delta["tool_calls"]:
      for tool_call in delta["tool_calls"]:
        if tool_call.get("type") == "function":
          yield FunctionChunk(
            id=tool_call.get("id"),
            name=tool_call.get("function", {}).get("name"),
            args=tool_call.get("function", {}).get("arguments"),
            index=tool_call.get("index", 0)
          ), finish_reason
  
  # Handle non-streaming message
  elif "message" in choice:
    message = choice["message"]
    
    if "content" in message and message["content"]:
      yield TextChunk(text=message["content"]), finish_reason
    
    if "tool_calls" in message and message["tool_calls"]:
      for tool_call in message["tool_calls"]:
        if tool_call.get("type") == "function":
          yield FunctionChunk(
            id=tool_call.get("id"),
            name=tool_call.get("function", {}).get("name"),
            args=tool_call.get("function", {}).get("arguments"),
            index=tool_call.get("index", 0)
          ), finish_reason
  
  # Handle usage metadata
  if "usage" in response:
    usage = response["usage"]
    yield UsageMetadataChunk(
      prompt_tokens=usage.get("prompt_tokens", 0),
      completion_tokens=usage.get("completion_tokens", 0),
      total_tokens=usage.get("total_tokens", 0)
    ), None
  
  # Yield finish reason if no content
  if finish_reason and "delta" not in choice and "message" not in choice:
    yield None, finish_reason


def _langdb_response_to_llm_response(response: Dict[str, Any]) -> LlmResponse:
  """Convert a LangDB response to LlmResponse.
  
  Args:
    response: The LangDB response.
    
  Returns:
    The LlmResponse.
  """
  if "choices" not in response or not response["choices"]:
    raise ValueError("No choices in response")
  
  choice = response["choices"][0]
  message = choice.get("message")
  
  if not message:
    raise ValueError("No message in response")
  
  parts = []
  
  if "content" in message and message["content"]:
    parts.append(types.Part.from_text(text=message["content"]))
  
  if "tool_calls" in message and message["tool_calls"]:
    for tool_call in message["tool_calls"]:
      if tool_call.get("type") == "function":
        part = types.Part.from_function_call(
          name=tool_call["function"]["name"],
          args=json.loads(tool_call["function"]["arguments"] or "{}")
        )
        part.function_call.id = tool_call.get("id", "")
        parts.append(part)
  
  llm_response = LlmResponse(
    content=types.Content(role="model", parts=parts)
  )
  
  # Add usage metadata if available
  if "usage" in response:
    usage = response["usage"]
    llm_response.usage_metadata = types.GenerateContentResponseUsageMetadata(
      prompt_token_count=usage.get("prompt_tokens", 0),
      candidates_token_count=usage.get("completion_tokens", 0),
      total_token_count=usage.get("total_tokens", 0)
    )
  
  return llm_response


def _message_to_llm_response(message: Dict[str, Any], is_partial: bool = False) -> LlmResponse:
  """Convert a message to LlmResponse.
  
  Args:
    message: The message to convert.
    is_partial: Whether the message is partial.
    
  Returns:
    The LlmResponse.
  """
  parts = []
  
  if "content" in message and message["content"]:
    parts.append(types.Part.from_text(text=message["content"]))
  
  if "tool_calls" in message and message["tool_calls"]:
    for tool_call in message["tool_calls"]:
      if tool_call.get("type") == "function":
        part = types.Part.from_function_call(
          name=tool_call["function"]["name"],
          args=json.loads(tool_call["function"]["arguments"] or "{}")
        )
        part.function_call.id = tool_call.get("id", "")
        parts.append(part)
  
  return LlmResponse(
    content=types.Content(role="model", parts=parts),
    partial=is_partial
  )


def _get_langdb_completion_inputs(llm_request: LlmRequest) -> Tuple[list, Optional[list]]:
  """Convert an LlmRequest to LangDB API format.
  
  Args:
    llm_request: The LlmRequest to convert.
    
  Returns:
    Tuple of (messages, tools).
  """
  messages = []
  
  # Add system instruction if present
  if llm_request.config.system_instruction:
    messages.append({
      "role": "system",
      "content": llm_request.config.system_instruction
    })
  
  # Convert content messages
  for content in llm_request.contents or []:
    message_or_list = _content_to_langdb_message(content)
    if isinstance(message_or_list, list):
      messages.extend(message_or_list)
    else:
      messages.append(message_or_list)
  
  # Convert tools
  tools = None
  if (llm_request.config and 
      llm_request.config.tools and 
      llm_request.config.tools[0].function_declarations):
    tools = [
      _function_declaration_to_langdb_tool(func_decl)
      for func_decl in llm_request.config.tools[0].function_declarations
    ]
  
  return messages, tools


class LangDBLlm(BaseLlm):
  """LangDB LLM implementation.
  
  This class provides integration with LangDB API, supporting both streaming 
  and non-streaming chat completions with function calling capabilities.
  
  Example usage:
  ```python
  langdb_llm = LangDBLlm(
      model="openai/gpt-4.1",
      api_key=os.getenv("LANGDB_API_KEY"),
      base_url=os.getenv("LANGDB_BASE_URL"),  # Optional
      project_id=os.getenv("LANGDB_PROJECT_ID"),
      extra_headers={
          "x-thread-id": "thread-123",
          "x-run-id": "run-456",
      }
  )
  ```
  
  Attributes:
    model: The model name (e.g., "openai/gpt-4.1").
    api_key: The API key for authentication.
    base_url: The base URL for LangDB API.
    project_id: The project ID.
    extra_headers: Additional headers to include in requests.
    client: The LangDB HTTP client.
  """
  
  api_key: str = Field(..., description="The API key for LangDB")
  base_url: str = Field(default=_DEFAULT_BASE_URL, description="The base URL for LangDB API")
  project_id: Optional[str] = Field(default=None, description="The project ID")
  extra_headers: Dict[str, str] = Field(default_factory=dict, description="Additional headers")
  client: LangDBClient = Field(default=None, description="The HTTP client")
  mcp_servers: Optional[list[Dict[str, Any]]] = Field(default=None, description="Remote MCP server configurations")
  
  def __init__(self, model_name: str, api_key: str, base_url: Optional[str] = None, 
               project_id: Optional[str] = None, extra_headers: Optional[Dict[str, str]] = None,
               mcp_servers: Optional[list[Dict[str, Any]]] = None, **kwargs):
    """Initialize LangDB LLM.
    
    Args:
      model_name: The name of the model.
      api_key: The API key for authentication.
      base_url: The base URL for LangDB API (optional).
      project_id: The project ID (optional).
      extra_headers: Additional headers to include in requests (optional).
      mcp_servers: Remote MCP server configurations (optional).
      **kwargs: Additional arguments.
    """
    # Handle project_id in headers
    headers = extra_headers or {}
    if project_id:
      headers["x-project-id"] = project_id
    
    # Initialize with corrected parameters
    super().__init__(
      model=model_name,
      api_key=api_key,
      base_url=base_url or _DEFAULT_BASE_URL,
      project_id=project_id,
      extra_headers=headers,
      mcp_servers=mcp_servers,
      **kwargs
    )
    
    # Create the client
    self.client = LangDBClient(
      api_key=self.api_key,
      base_url=self.base_url,
      extra_headers=self.extra_headers
    )
  
  def _is_remote_mcp_tool(self, function_name: str) -> bool:
    """Check if a function name indicates a remote MCP tool.
    
    Args:
      function_name: The name of the function to check.
      
    Returns:
      True if this appears to be a remote MCP tool name.
    """
    # Common patterns for remote MCP tool names
    mcp_patterns = [
      '-mcp---',     # tavily-mcp---tavily-search
      'mcp_',        # mcp_search
      '_mcp_',       # tool_mcp_search
      'mcp-',        # mcp-tool-name
      '-mcp-',       # tool-mcp-search
    ]
    
    function_name_lower = function_name.lower()
    return any(pattern in function_name_lower for pattern in mcp_patterns)
  
  def _filter_remote_mcp_function_calls(self, content: types.Content) -> types.Content:
    """Filter out remote MCP function calls from content to prevent local execution.
    
    When MCP servers are configured, LangDB executes MCP tools remotely but still
    returns function call responses. We need to filter these out to prevent ADK
    from trying to execute them locally.
    
    Args:
      content: The content that may contain function calls.
      
    Returns:
      Content with remote MCP function calls filtered out.
    """
    if not self.mcp_servers or not content.parts:
      return content  # No MCP servers or no parts to filter
    
    filtered_parts = []
    for part in content.parts:
      if part.function_call and self._is_remote_mcp_tool(part.function_call.name):
        # Skip remote MCP function calls - they're executed on LangDB
        logger.debug(f"Filtered remote MCP tool call: {part.function_call.name}")
        continue
      else:
        # Keep local function calls and text parts
        filtered_parts.append(part)
    
    # Return content with filtered parts
    return types.Content(role=content.role, parts=filtered_parts)

  @override
  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    """Generate content asynchronously.
    
    Args:
      llm_request: The request to send to LangDB.
      stream: Whether to stream the response.
      
    Yields:
      LlmResponse objects.
    """
    self._maybe_append_user_content(llm_request)
    
    messages, tools = _get_langdb_completion_inputs(llm_request)
    
    payload = {
      "model": self.model,
      "messages": messages
    }
    
    if tools:
      payload["tools"] = tools
    
    # Add MCP servers from instance configuration (auto-configured by Agent)
    if self.mcp_servers:
      payload["mcp_servers"] = self.mcp_servers
    
    try:
      if stream:
        response_generator = await self.client.chat_completion(payload, stream=True)
        text = ""
        function_calls = {}
        aggregated_response = None
        aggregated_tool_response = None
        usage_metadata = None
        
        for response in response_generator:
          for chunk, finish_reason in _langdb_response_to_chunk(response):
            if isinstance(chunk, FunctionChunk):
              index = chunk.index or 0
              if index not in function_calls:
                function_calls[index] = {"name": "", "args": "", "id": None}
              
              if chunk.name:
                function_calls[index]["name"] += chunk.name
              if chunk.args:
                function_calls[index]["args"] += chunk.args
              if chunk.id:
                function_calls[index]["id"] = chunk.id
            
            elif isinstance(chunk, TextChunk):
              text += chunk.text
              yield _message_to_llm_response(
                {"content": chunk.text}, is_partial=True
              )
            
            elif isinstance(chunk, UsageMetadataChunk):
              usage_metadata = types.GenerateContentResponseUsageMetadata(
                prompt_token_count=chunk.prompt_tokens,
                candidates_token_count=chunk.completion_tokens,
                total_token_count=chunk.total_tokens
              )
            
            if finish_reason in ["tool_calls", "stop"] and function_calls:
              tool_calls = []
              for index, func_data in function_calls.items():
                if func_data["id"]:
                  tool_calls.append({
                    "id": func_data["id"],
                    "type": "function",
                    "function": {
                      "name": func_data["name"],
                      "arguments": func_data["args"]
                    }
                  })
              
              aggregated_tool_response = _message_to_llm_response(
                {"tool_calls": tool_calls}
              )
              function_calls.clear()
            
            elif finish_reason == "stop" and text:
              aggregated_response = _message_to_llm_response(
                {"content": text}
              )
              text = ""
        
        # Yield final responses with usage metadata, filtering remote MCP tools
        if aggregated_response:
          if usage_metadata:
            aggregated_response.usage_metadata = usage_metadata
          # Filter remote MCP function calls from response
          aggregated_response.content = self._filter_remote_mcp_function_calls(aggregated_response.content)
          yield aggregated_response
        
        if aggregated_tool_response:
          if usage_metadata and not aggregated_response:
            aggregated_tool_response.usage_metadata = usage_metadata
          # Filter remote MCP function calls from tool response
          aggregated_tool_response.content = self._filter_remote_mcp_function_calls(aggregated_tool_response.content)
          # Only yield if there are still function calls after filtering
          if aggregated_tool_response.content.parts:
            yield aggregated_tool_response
      
      else:
        response = await self.client.chat_completion(payload, stream=False)
        llm_response = _langdb_response_to_llm_response(response)
        # Filter remote MCP function calls from non-streaming response
        llm_response.content = self._filter_remote_mcp_function_calls(llm_response.content)
        yield llm_response
    
    except Exception as e:
      logger.error(f"Error in LangDB API call: {e}")
      raise
  
  @staticmethod
  @override
  def supported_models() -> list[str]:
    """Return supported models.
    
    Returns:
      List of supported model patterns.
    """
    return [".*"]  # LangDB supports various models through providers


# Factory function for convenience
def langdb_llm(model_name: str, api_key: str, base_url: Optional[str] = None,
               project_id: Optional[str] = None, extra_headers: Optional[Dict[str, str]] = None,
               mcp_servers: Optional[list[Dict[str, Any]]] = None) -> LangDBLlm:
  """Factory function to create a LangDB LLM instance.
  
  Args:
    model_name: The name of the model.
    api_key: The API key for authentication.
    base_url: The base URL for LangDB API (optional).
    project_id: The project ID (optional).
    extra_headers: Additional headers to include in requests (optional).
    mcp_servers: Remote MCP server configurations (optional).
    
  Returns:
    A configured LangDBLlm instance.
  """
  return LangDBLlm(
    model_name=model_name,
    api_key=api_key,
    base_url=base_url,
    project_id=project_id,
    extra_headers=extra_headers,
    mcp_servers=mcp_servers
  )
