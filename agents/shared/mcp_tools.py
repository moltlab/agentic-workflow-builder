"""
MCP resource helper tools (LangChain BaseTool) for use with LangGraph agents.
"""

from typing import List

from langchain_core.tools import BaseTool

from agents.mcp_client import MCPClient
from utils.logging_utils import get_logger

logger = get_logger("mcp_tools")


class ResourceAccessTool(BaseTool):
    """Read a resource URI from the MCP server."""

    name: str = "resource_reader"
    description: str = (
        "Reads content from a specified resource on the MCP server. "
        "Use this to access files or data lists. The input must be the full resource URI, "
        "for example: 'file:///shared_files/report.pdf' or 'resource://files/list'."
    )
    mcp_client: MCPClient

    def _run(self, uri: str) -> str:
        try:
            return self.mcp_client.read_resource_text(uri=uri)
        except Exception as e:
            logger.error(f"ResourceAccessTool failed for URI '{uri}': {e}")
            return f"An error occurred while trying to read the resource: {e}"


class ListFilesTool(BaseTool):
    """List project files from the MCP server."""

    name: str = "list_files"
    description: str = (
        "Lists all available project files from the MCP server's shared resources. "
        "Use this to see what files are accessible to the agent. No input required."
    )
    mcp_client: MCPClient

    def _run(self) -> str:
        try:
            files = self.mcp_client.list_project_files()
            file_list = "\n".join(files) if files else "None"
            return f"Available files: {file_list}"
        except Exception as e:
            logger.error(f"ListFilesTool failed: {e}")
            return f"An error occurred while trying to list files: {e}"


class SummarizeTextTool(BaseTool):
    """Summarize text via MCP."""

    name: str = "summarize_text"
    description: str = (
        "Summarizes a given text using an MCP prompt. "
        "Input should be the text content to be summarized."
    )
    mcp_client: MCPClient

    def _run(self, text: str) -> str:
        try:
            return self.mcp_client.call_tool("summarize_text", text=text)
        except Exception as e:
            logger.error(f"SummarizeTextTool failed: {e}")
            return f"An error occurred while trying to summarize text: {e}"


class FindKeywordsTool(BaseTool):
    """Keyword extraction via MCP."""

    name: str = "find_keywords"
    description: str = (
        "Extracts keywords from a given text using an MCP prompt. "
        "Input should be the text content from which to extract keywords."
    )
    mcp_client: MCPClient

    def _run(self, text: str) -> str:
        try:
            return self.mcp_client.call_tool("find_keywords", text=text)
        except Exception as e:
            logger.error(f"FindKeywordsTool failed: {e}")
            return f"An error occurred while trying to find keywords: {e}"


def create_mcp_resource_tools(mcp_client: MCPClient) -> List[BaseTool]:
    return [
        ResourceAccessTool(mcp_client=mcp_client),
        ListFilesTool(mcp_client=mcp_client),
        SummarizeTextTool(mcp_client=mcp_client),
        FindKeywordsTool(mcp_client=mcp_client),
    ]
