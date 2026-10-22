"""Bearer-token check. The MCP SDK calls `verify_token` for every HTTP request and answers
401 itself when it returns None, before any tool runs."""

import hmac

from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import AccessToken

from news_mcp import config


class StaticTokenVerifier:
    async def verify_token(self, token: str) -> AccessToken:
        """
        :param token: the value after "Authorization: Bearer"
        :return: the token's identity and scopes, or None if the token is unknown
        """
        for known_token, policy in config.TOKENS.items():
            if hmac.compare_digest(token.encode(), known_token.encode()):  # constant-time comparison
                return AccessToken(token=token, client_id=policy["client_id"], scopes=policy["scopes"])
        return None


def caller_has_scope(scope: str) -> bool:
    """True if the token of the current request has this scope."""
    token = get_access_token()
    return token is not None and scope in token.scopes
