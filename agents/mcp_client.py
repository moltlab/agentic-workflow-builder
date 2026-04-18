import json
import httpx
import time
import threading
from typing import Any, Dict, Optional
from urllib.parse import urljoin, urlparse, parse_qs
from utils.logging_utils import get_logger

logger = get_logger('mcp_client')

class MCPClient:
    def __init__(self, sse_url: str, server_base_url: str) -> None:
        self.sse_url = sse_url
        self.server_base_url = server_base_url
        self.post_url: Optional[str] = None
        self.session_id: Optional[str] = None
        self._client = httpx.Client()
        self._responses: Dict[int, Dict[str, Any]] = {}
        self._cond = threading.Condition()
        self._sse_thread: Optional[threading.Thread] = None
        self._stop = False

    def _sse_loop(self) -> None:
        headers = {"Accept": "text/event-stream", "Cache-Control": "no-cache"}
        try:
            with self._client.stream("GET", self.sse_url, headers=headers, timeout=60) as response:
                current_event: Dict[str, str] = {}
                for raw_line in response.iter_lines():
                    if self._stop:
                        break
                    line = raw_line.strip()
                    if not line:
                        if 'event' in current_event and 'data' in current_event:
                            evt = current_event['event']
                            data = current_event['data']
                            if evt == 'endpoint':
                                rel = data
                                absolute = urljoin(self.server_base_url, rel)
                                parsed = urlparse(absolute)
                                q = parse_qs(parsed.query)
                                sid = q.get('session_id', [])
                                if sid:
                                    self.session_id = sid[0]
                                self.post_url = absolute
                                with self._cond:
                                    self._cond.notify_all()
                            elif evt == 'message':
                                try:
                                    msg = json.loads(data)
                                except Exception:
                                    msg = {"malformed": data}
                                if isinstance(msg, dict) and 'id' in msg:
                                    with self._cond:
                                        self._responses[msg['id']] = msg
                                        self._cond.notify_all()
                        current_event = {}
                        continue
                    if line.startswith('event:'):
                        current_event['event'] = line[len('event: '):].strip()
                    elif line.startswith('data:'):
                        current_event['data'] = line[len('data: '):].strip()
        except Exception:
            pass

    def connect(self) -> None:
        if self._sse_thread and self._sse_thread.is_alive():
            return
        self._stop = False
        self._sse_thread = threading.Thread(target=self._sse_loop, daemon=True)
        self._sse_thread.start()
        t0 = time.time()
        while time.time() - t0 < 5:
            if self.post_url:
                return
            time.sleep(0.05)
        raise RuntimeError("Failed to get MCP post URL from SSE endpoint")

    def stop(self):
        self._stop = True
        if self._sse_thread and self._sse_thread.is_alive():
            # Give a small window for the SSE thread to pick up the stop signal
            self._sse_thread.join(timeout=1)
        self._client.close()
        logger.info("MCP Client stopped and HTTP client closed.")

    def post(self, payload: Dict[str, Any]) -> int:
        if not self.post_url:
            raise RuntimeError("No MCP post URL available")
        req_id = payload.get('id')
        r = self._client.post(self.post_url, json=payload, timeout=15, follow_redirects=True)
        r.raise_for_status()
        return int(req_id) if req_id is not None else -1

    def rpc(self, method: str, params: Dict[str, Any] | None = None, timeout: float = 10.0) -> Dict[str, Any]:
        req_id = int(time.time() * 1000) % 2_147_483_647
        payload: Dict[str, Any] = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params is not None:
            payload["params"] = params
        self.post(payload)
        deadline = time.time() + timeout
        with self._cond:
            while time.time() < deadline:
                if req_id in self._responses:
                    return self._responses.pop(req_id)
                remaining = deadline - time.time()
                if remaining > 0:
                    self._cond.wait(timeout=min(0.2, remaining))
        raise TimeoutError(f"Timed out waiting for response to {method}")

    def initialize(self) -> None:
        self.rpc(
            "initialize",
            {
                "protocolVersion": 1,
                "clientInfo": {"name": "agent", "version": "0.1.0"},
                "capabilities": {},
            },
            timeout=10.0,
        )
        self.post({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def call_tool(self, name: str, **kwargs: Any) -> str:
        resp = self.rpc("tools/call", {"name": name, "arguments": kwargs}, timeout=20.0)
        if 'error' in resp:
            return json.dumps(resp['error'])
        result = resp.get('result', {})
        return json.dumps(result)

    def read_resource(self, uri: str) -> str:
        resp = self.rpc("resources/read", {"uri": uri}, timeout=10.0)
        if 'error' in resp:
            return json.dumps(resp['error'])
        result = resp.get('result', {})
        contents = result.get('contents', [])
        parts: list[str] = []
        for c in contents:
            if 'text' in c and c['text'] is not None:
                parts.append(str(c['text']))
            elif 'blob' in c and c['blob'] is not None:
                parts.append(f"<binary content {len(c['blob'])} base64 chars>")
        return "\n".join(parts) if parts else json.dumps(result)

    def read_resource_text(self, uri: str) -> str:
        resp = self.rpc("resources/read", {"uri": uri}, timeout=10.0)
        if 'error' in resp:
            return ""
        result = resp.get('result', {})
        for c in result.get('contents', []):
            text = c.get('text')
            if text:
                return str(text)
        return ""

    def list_project_files(self) -> list[str]:
        try:
            resp = self.rpc("resources/read", {"uri": "resource://files/list"}, timeout=10.0)
            if 'error' in resp:
                logger.error(f"Error listing project files from MCP: {resp['error']}")
                return []
            result = resp.get('result', {})
            all_files: list[str] = []
            for c in result.get('contents', []):
                text = c.get('text')
                if text:
                    directory_files = json.loads(text)
                    if isinstance(directory_files, dict):
                        for alias, files in directory_files.items():
                            if isinstance(files, list):
                                # Prepend alias to filename for clear identification
                                all_files.extend([f"{alias}/{f}" for f in files])
                    elif isinstance(directory_files, list): # Fallback for old format if necessary
                        all_files.extend([str(x) for x in directory_files])
            return all_files
        except Exception as e:
            logger.error(f"Exception when parsing project files from MCP: {e}")
            return []