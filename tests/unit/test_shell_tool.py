"""Testa a logica de propose/confirm/expiracao isoladamente -- sem rodar
nenhum comando de verdade e sem precisar do servidor MCP no ar."""
from __future__ import annotations

import time

from persona.tools.shell_tool import PendingCommands


def test_confirm_returns_the_proposed_command():
    pending = PendingCommands(ttl_s=60)
    token = pending.propose("echo oi")

    assert pending.confirm(token) == "echo oi"


def test_confirm_with_unknown_token_returns_none():
    pending = PendingCommands(ttl_s=60)
    assert pending.confirm("token-que-nao-existe") is None


def test_confirm_is_single_use():
    pending = PendingCommands(ttl_s=60)
    token = pending.propose("echo oi")

    assert pending.confirm(token) == "echo oi"
    assert pending.confirm(token) is None  # segunda tentativa com o mesmo token falha


def test_confirm_fails_after_ttl_expires():
    pending = PendingCommands(ttl_s=0.05)
    token = pending.propose("echo oi")

    time.sleep(0.1)

    assert pending.confirm(token) is None


def test_two_proposals_get_different_tokens_and_are_independent():
    pending = PendingCommands(ttl_s=60)
    token_a = pending.propose("comando A")
    token_b = pending.propose("comando B")

    assert token_a != token_b
    assert pending.confirm(token_a) == "comando A"
    assert pending.confirm(token_b) == "comando B"


def test_pending_count_reflects_state():
    pending = PendingCommands(ttl_s=60)
    assert pending.pending_count() == 0

    token = pending.propose("echo oi")
    assert pending.pending_count() == 1

    pending.confirm(token)
    assert pending.pending_count() == 0
