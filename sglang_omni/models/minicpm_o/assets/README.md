# Summary instruction audio

`summary_instruction.wav` says: “Please recap our conversation, keeping as much detail as possible.” It is a 16 kHz mono PCM16 WAV generated with macOS Samantha at 175 words per minute. Its duration is about 4.27 seconds. This is input audio for native duplex context compaction; the compaction does not synthesize or save output speech.

Set `window.summary_audio_path` explicitly to this file or to a deployment-provided recording of the same instruction. The server does not depend on a personal downloads directory or on macOS speech synthesis at runtime.
