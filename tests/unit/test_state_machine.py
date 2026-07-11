from persona.bus.events import STTFinal, TTSAudioChunkReady, VADSpeechEnd, VADSpeechStart
from persona.coordinator.state_machine import (
    EMPTY_UTTERANCE,
    PLAYBACK_NATURALLY_STOPPED,
    SideEffect,
    TurnState,
    TurnStateMachine,
)


def test_idle_to_user_speaking_on_vad_start():
    fsm = TurnStateMachine()
    transition = fsm.handle(VADSpeechStart())

    assert transition.new_state is TurnState.USER_SPEAKING
    assert SideEffect.BEGIN_UTTERANCE in transition.side_effects
    assert SideEffect.CANCEL_IDLE_TIMER in transition.side_effects
    assert not transition.is_barge_in
    assert fsm.state is TurnState.USER_SPEAKING


def test_user_speaking_to_transcribing_on_vad_end():
    fsm = TurnStateMachine()
    fsm.handle(VADSpeechStart())
    transition = fsm.handle(VADSpeechEnd(duration_s=1.2))

    assert transition.new_state is TurnState.TRANSCRIBING
    assert transition.side_effects == (SideEffect.FINALIZE_UTTERANCE,)


def test_transcribing_to_thinking_on_stt_final():
    fsm = TurnStateMachine()
    fsm.handle(VADSpeechStart())
    fsm.handle(VADSpeechEnd(duration_s=1.2))
    transition = fsm.handle(STTFinal(utterance_id="u1", text="oi", duration_s=1.2))

    assert transition.new_state is TurnState.THINKING
    assert transition.side_effects == (SideEffect.DISPATCH_LLM,)


def test_transcribing_empty_utterance_returns_to_idle():
    fsm = TurnStateMachine()
    fsm.handle(VADSpeechStart())
    fsm.handle(VADSpeechEnd(duration_s=0.3))
    transition = fsm.handle(EMPTY_UTTERANCE)

    assert transition.new_state is TurnState.IDLE
    assert SideEffect.START_IDLE_TIMER in transition.side_effects


def test_thinking_to_speaking_on_first_audio_chunk():
    fsm = TurnStateMachine()
    fsm.handle(VADSpeechStart())
    fsm.handle(VADSpeechEnd(duration_s=1.0))
    fsm.handle(STTFinal(utterance_id="u1", text="oi", duration_s=1.0))
    transition = fsm.handle(
        TTSAudioChunkReady(turn_id="t1", sentence_id=1, pcm=None, sample_rate=24000)
    )

    assert transition.new_state is TurnState.SPEAKING
    assert transition.side_effects == (SideEffect.PLAY_CHUNK,)


def test_barge_in_while_speaking():
    fsm = TurnStateMachine()
    fsm.handle(VADSpeechStart())
    fsm.handle(VADSpeechEnd(duration_s=1.0))
    fsm.handle(STTFinal(utterance_id="u1", text="oi", duration_s=1.0))
    fsm.handle(TTSAudioChunkReady(turn_id="t1", sentence_id=1, pcm=None, sample_rate=24000))

    transition = fsm.handle(VADSpeechStart())

    assert transition.new_state is TurnState.USER_SPEAKING
    assert transition.is_barge_in
    assert transition.side_effects == (
        SideEffect.STOP_PLAYBACK,
        SideEffect.CANCEL_LLM,
        SideEffect.CANCEL_TTS,
        SideEffect.BEGIN_UTTERANCE,
    )


def test_barge_in_while_thinking_before_any_audio():
    fsm = TurnStateMachine()
    fsm.handle(VADSpeechStart())
    fsm.handle(VADSpeechEnd(duration_s=1.0))
    fsm.handle(STTFinal(utterance_id="u1", text="oi", duration_s=1.0))

    transition = fsm.handle(VADSpeechStart())

    assert transition.new_state is TurnState.USER_SPEAKING
    assert transition.is_barge_in


def test_speaking_to_idle_on_natural_playback_stop():
    fsm = TurnStateMachine()
    fsm.handle(VADSpeechStart())
    fsm.handle(VADSpeechEnd(duration_s=1.0))
    fsm.handle(STTFinal(utterance_id="u1", text="oi", duration_s=1.0))
    fsm.handle(TTSAudioChunkReady(turn_id="t1", sentence_id=1, pcm=None, sample_rate=24000))

    transition = fsm.handle(PLAYBACK_NATURALLY_STOPPED)

    assert transition.new_state is TurnState.IDLE
    assert SideEffect.START_IDLE_TIMER in transition.side_effects


def test_unmapped_event_is_a_noop():
    fsm = TurnStateMachine()
    transition = fsm.handle(VADSpeechEnd(duration_s=0.5))  # IDLE nao mapeia VADSpeechEnd

    assert transition.new_state is TurnState.IDLE
    assert transition.side_effects == ()
    assert fsm.state is TurnState.IDLE


def test_barge_in_latency_is_measurable():
    fsm = TurnStateMachine()
    fsm.handle(VADSpeechStart())
    fsm.handle(VADSpeechEnd(duration_s=1.0))
    fsm.handle(STTFinal(utterance_id="u1", text="oi", duration_s=1.0))
    fsm.handle(TTSAudioChunkReady(turn_id="t1", sentence_id=1, pcm=None, sample_rate=24000))

    assert fsm.last_barge_in_latency_s() is None

    fsm.handle(VADSpeechStart())  # este e o barge-in

    latency = fsm.last_barge_in_latency_s()
    assert latency is not None
    assert 0 <= latency < 1.0  # bem abaixo da meta de 100ms num teste sem I/O real
