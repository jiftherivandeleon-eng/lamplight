"""Build check: the speech decoder must really decode a sound. A version mismatch between faster-whisper and PyAV fails here, not on a student's PC."""
import math, os, struct, sys, tempfile, wave

try:
    from faster_whisper.audio import decode_audio
    path = os.path.join(tempfile.gettempdir(), "lamplight_check.wav")
    w = wave.open(path, "wb")
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
    w.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(i / 20))) for i in range(16000)))
    w.close()
    n = len(decode_audio(path, sampling_rate=16000))
    if n < 10000:
        sys.exit("Speech decoder returned too little audio (%d samples)." % n)
    print("Speech decoder OK (%d samples)" % n)
except Exception as e:
    sys.exit("Speech decoder check FAILED: %r" % (e,))
