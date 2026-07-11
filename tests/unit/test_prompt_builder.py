from persona.llm.prompt_builder import PromptContext, build_messages


def test_minimal_context_produces_system_and_user_messages():
    ctx = PromptContext(system_prompt="Voce e o Persona.", user_utterance="Oi, tudo bem?")
    messages = build_messages(ctx)

    assert messages == [
        {"role": "system", "content": "Voce e o Persona."},
        {"role": "user", "content": "Oi, tudo bem?"},
    ]


def test_memory_block_is_appended_to_system_message():
    ctx = PromptContext(
        system_prompt="Voce e o Persona.",
        user_utterance="O que eu estava fazendo?",
        profile_summary="- nome: Mikael",
        episodic_snippets=["Falamos sobre o projeto Persona ontem."],
    )
    messages = build_messages(ctx)

    system_message = messages[0]["content"]
    assert "Voce e o Persona." in system_message
    assert "nome: Mikael" in system_message
    assert "Falamos sobre o projeto Persona ontem." in system_message


def test_short_term_turns_are_preserved_in_order():
    ctx = PromptContext(
        system_prompt="sys",
        user_utterance="e agora?",
        short_term_turns=[("user", "primeira pergunta"), ("assistant", "primeira resposta")],
    )
    messages = build_messages(ctx)

    assert messages == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "primeira pergunta"},
        {"role": "assistant", "content": "primeira resposta"},
        {"role": "user", "content": "e agora?"},
    ]


def test_oldest_short_term_turns_are_dropped_first_when_over_budget():
    turns = [("user", "a" * 100), ("assistant", "b" * 100), ("user", "c" * 20)]
    ctx = PromptContext(
        system_prompt="sys",
        user_utterance="final",
        short_term_turns=turns,
        max_tokens=30,  # ~120 chars a 4 chars/token: forca descartar os turnos mais antigos
    )
    messages = build_messages(ctx)

    contents = [m["content"] for m in messages]
    assert "a" * 100 not in contents
    assert "c" * 20 in contents


def test_episodic_snippets_trimmed_after_short_term_exhausted():
    ctx = PromptContext(
        system_prompt="sys",
        user_utterance="final",
        episodic_snippets=["primeiro trecho relevante " * 20, "segundo trecho " * 2],
        max_tokens=15,
    )
    messages = build_messages(ctx)

    system_message = messages[0]["content"]
    assert "primeiro trecho relevante" not in system_message
