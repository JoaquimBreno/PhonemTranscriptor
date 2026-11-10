__all__ = ["transcribe"]


def __getattr__(name: str):
    # Lazy so the GAME worker can import submodules without loading Whisper/SOFA.
    if name == "transcribe":
        from phonem_transcriptor.pipeline import transcribe

        return transcribe
    raise AttributeError(name)
