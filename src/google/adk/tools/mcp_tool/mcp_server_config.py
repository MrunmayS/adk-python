# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration models for remote MCP servers."""

from __future__ import annotations

from typing import Dict, Optional, Any
from pydantic import BaseModel, Field


class MCPServerConfig(BaseModel):
    """Configuration for a remote MCP server.
    
    This class defines the configuration needed to connect to a remote
    MCP server that is hosted on LangDB's infrastructure.
    
    Example usage:
    ```python
    server_config = MCPServerConfig(
        server_url="https://api.staging.langdb.ai/duckduck_og0eufki",
        type="sse",
        name="duckduckgo_search",
        description="DuckDuckGo search capabilities"
    )
    ```
    """
    
    server_url: str = Field(..., description="URL of the remote MCP server")
    type: str = Field(..., description="Connection type (e.g., 'sse', 'http', 'websocket')")
    name: Optional[str] = Field(default=None, description="Human-readable name for the MCP server")
    description: Optional[str] = Field(default=None, description="Description of the MCP server's capabilities")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata for the server")
    
    def to_langdb_format(self) -> Dict[str, Any]:
        """Convert to LangDB API format.
        
        Returns:
            Dictionary in the format expected by LangDB's mcp_servers field.
        """
        result = {
            "server_url": self.server_url,
            "type": self.type
        }
        
        # Add optional fields if present
        if self.name:
            result["name"] = self.name
        if self.description:
            result["description"] = self.description
        if self.metadata:
            result.update(self.metadata)
            
        return result
    
    @classmethod
    def from_url(cls, server_url: str, server_type: str = "sse", **kwargs) -> MCPServerConfig:
        """Create MCP server config from URL and type.
        
        Args:
            server_url: URL of the remote MCP server.
            server_type: Connection type (defaults to "sse").
            **kwargs: Additional configuration options.
            
        Returns:
            MCPServerConfig instance.
        """
        return cls(
            server_url=server_url,
            type=server_type,
            **kwargs
        )


class RemoteMCPConfig(BaseModel):
    """Configuration for multiple remote MCP servers.
    
    This class manages a collection of MCP server configurations
    and provides utilities for working with them.
    """
    
    servers: list[MCPServerConfig] = Field(default_factory=list, description="List of MCP server configurations")
    
    def add_server(self, server: MCPServerConfig) -> None:
        """Add an MCP server configuration.
        
        Args:
            server: MCP server configuration to add.
        """
        self.servers.append(server)
    
    def add_server_url(self, server_url: str, server_type: str = "sse", **kwargs) -> None:
        """Add MCP server by URL.
        
        Args:
            server_url: URL of the remote MCP server.
            server_type: Connection type (defaults to "sse").
            **kwargs: Additional configuration options.
        """
        server = MCPServerConfig.from_url(server_url, server_type, **kwargs)
        self.add_server(server)
    
    def to_langdb_format(self) -> list[Dict[str, Any]]:
        """Convert all servers to LangDB API format.
        
        Returns:
            List of dictionaries in the format expected by LangDB's mcp_servers field.
        """
        return [server.to_langdb_format() for server in self.servers]
    
    def __len__(self) -> int:
        """Return number of configured servers."""
        return len(self.servers)
    
    def __bool__(self) -> bool:
        """Return True if any servers are configured."""
        return len(self.servers) > 0