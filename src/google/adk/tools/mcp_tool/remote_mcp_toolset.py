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

"""Remote MCP Toolset for LangDB-hosted MCP servers."""

from __future__ import annotations

import logging
from typing import List, Optional, Union

from ...agents.readonly_context import ReadonlyContext
from ..base_tool import BaseTool
from ..base_toolset import BaseToolset, ToolPredicate
from .mcp_server_config import MCPServerConfig, RemoteMCPConfig

logger = logging.getLogger("google_adk." + __name__)


class RemoteMCPToolset(BaseToolset):
    """Toolset for remote MCP servers hosted on LangDB.
    
    Unlike the local MCPToolset, this toolset doesn't establish local
    connections to MCP servers. Instead, it configures the LangDB LLM
    to use remote MCP servers hosted on LangDB's infrastructure.
    
    Usage:
    ```python
    # Single server
    toolset = RemoteMCPToolset(
        server_url="https://api.staging.langdb.ai/duckduck_og0eufki",
        server_type="sse"
    )
    
    # Multiple servers
    toolset = RemoteMCPToolset()
    toolset.add_server(
        server_url="https://api.staging.langdb.ai/duckduck_og0eufki",
        server_type="sse",
        name="duckduckgo_search"
    )
    toolset.add_server(
        server_url="https://api.staging.langdb.ai/filesystem_server",
        server_type="sse",
        name="filesystem"
    )
    
    # Use in agent
    agent = LlmAgent(
        model=langdb_llm(...),  # Must use LangDB LLM
        name='remote_mcp_agent',
        instruction='Help users with search and file operations',
        tools=[toolset],
    )
    ```
    """
    
    def __init__(
        self,
        *,
        server_url: Optional[str] = None,
        server_type: str = "sse",
        server_name: Optional[str] = None,
        server_description: Optional[str] = None,
        mcp_config: Optional[RemoteMCPConfig] = None,
        tool_filter: Optional[Union[ToolPredicate, List[str]]] = None,
    ):
        """Initialize RemoteMCPToolset.
        
        Args:
            server_url: URL of a single MCP server (optional).
            server_type: Connection type for the server (default: "sse").
            server_name: Human-readable name for the server (optional).
            server_description: Description of the server's capabilities (optional).
            mcp_config: Pre-configured RemoteMCPConfig (optional).
            tool_filter: Tool filtering (not used for remote MCP but kept for interface compatibility).
        """
        super().__init__(tool_filter=tool_filter)
        
        if mcp_config:
            self.mcp_config = mcp_config
        else:
            self.mcp_config = RemoteMCPConfig()
            
        # Add initial server if provided
        if server_url:
            self.add_server(
                server_url=server_url,
                server_type=server_type,
                name=server_name,
                description=server_description
            )
    
    def add_server(
        self,
        server_url: str,
        server_type: str = "sse",
        name: Optional[str] = None,
        description: Optional[str] = None,
        **metadata
    ) -> None:
        """Add an MCP server to the toolset.
        
        Args:
            server_url: URL of the remote MCP server.
            server_type: Connection type (e.g., "sse", "http").
            name: Human-readable name for the server.
            description: Description of the server's capabilities.
            **metadata: Additional metadata for the server.
        """
        server_config = MCPServerConfig(
            server_url=server_url,
            type=server_type,
            name=name,
            description=description,
            metadata=metadata
        )
        self.mcp_config.add_server(server_config)
        logger.info(f"Added remote MCP server: {server_url} (type: {server_type})")
    
    def add_server_config(self, server_config: MCPServerConfig) -> None:
        """Add a pre-configured MCP server.
        
        Args:
            server_config: MCP server configuration to add.
        """
        self.mcp_config.add_server(server_config)
        logger.info(f"Added remote MCP server: {server_config.server_url}")
    
    async def get_tools(
        self,
        readonly_context: Optional[ReadonlyContext] = None,
    ) -> list[BaseTool]:
        """Return empty list for remote MCP servers.
        
        For remote MCP toolsets, tools are executed on LangDB infrastructure,
        not locally. Therefore, we return an empty list to avoid conflicts
        with local tool execution while still allowing the MCP servers to
        be configured in the LangDB payload.
        
        Args:
            readonly_context: Context (not used for remote MCP).
            
        Returns:
            Empty list (tools are handled remotely by LangDB).
        """
        # Remote MCP servers are handled by LangDB, not locally
        # Return empty list to avoid local tool conflicts
        logger.info(f"RemoteMCPToolset configured with {len(self.mcp_config.servers)} remote servers")
        return []
    
    async def close(self) -> None:
        """Cleanup resources.
        
        For remote MCP toolsets, there are no local connections to close,
        so this method is a no-op.
        """
        logger.debug("Closing RemoteMCPToolset (no-op for remote servers)")
        pass
    
    def get_mcp_servers_config(self) -> list[dict]:
        """Get MCP server configurations for LangDB API.
        
        Returns:
            List of server configurations in LangDB API format.
        """
        return self.mcp_config.to_langdb_format()
    
    def __len__(self) -> int:
        """Return number of configured MCP servers."""
        return len(self.mcp_config)
    
    def __bool__(self) -> bool:
        """Return True if any MCP servers are configured."""
        return bool(self.mcp_config)