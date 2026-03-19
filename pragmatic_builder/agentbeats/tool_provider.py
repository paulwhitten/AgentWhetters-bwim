import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from agentbeats.client import send_message, DEFAULT_TIMEOUT


class ToolProvider:
    def __init__(self):
        self._context_ids = {}
        self._a2a_clients = {}
        self._httpx_clients = {}

    async def _get_a2a_client(self, url: str):
        if url not in self._a2a_clients:
            httpx_client = httpx.AsyncClient(timeout=DEFAULT_TIMEOUT)
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=url)
            agent_card = await resolver.get_agent_card()
            config = ClientConfig(httpx_client=httpx_client, streaming=True)
            self._a2a_clients[url] = ClientFactory(config).create(agent_card)
            self._httpx_clients[url] = httpx_client
        return self._a2a_clients[url]

    async def talk_to_agent(self, message: str, url: str, new_conversation: bool = False):
        """
        Communicate with another agent by sending a message and receiving their response.

        Args:
            message: The message to send to the agent
            url: The agent's URL endpoint
            new_conversation: If True, start fresh conversation; if False, continue existing conversation

        Returns:
            str: The agent's response message
        """
        a2a_client = await self._get_a2a_client(url)
        outputs = await send_message(
            message=message,
            base_url=url,
            context_id=None if new_conversation else self._context_ids.get(url, None),
            streaming=True,
            a2a_client=a2a_client,
        )
        if outputs.get("status", "completed") != "completed":
            raise RuntimeError(f"{url} responded with: {outputs}")
        self._context_ids[url] = outputs.get("context_id", None)
        return outputs["response"]

    async def reset(self):
        for httpx_client in self._httpx_clients.values():
            await httpx_client.aclose()
        self._context_ids = {}
        self._a2a_clients = {}
        self._httpx_clients = {}
