# Lamplight

A private, offline AI study workspace. Chat with your own notes, PDFs, slides, pictures and voice recordings, and build study guides from them. It runs on your own computer with [Ollama](https://ollama.com). Nothing is uploaded unless you choose Cloud.

## Features
- **Ask your documents.** Select a document and ask questions. Answers say where they came from, and general chat works too.
- **Study guides** with an answer key, built from a whole document.
- **Pictures and voice.** Pictures are read by the local vision model. Recordings are transcribed offline with Whisper, with time markers.
- **Memory.** Chat history, saved facts and search over past chats, all kept on your PC.
- **Local or Cloud.** Local is private and offline. Cloud (optional) needs a free Lamplight login stored on your PC and an Ollama sign-in.
- Works in small windows, with a menu on phones and narrow screens.

## Quick start
1. Install [Python 3.10+](https://www.python.org/downloads/) (tick *Add python.exe to PATH*) and [Ollama](https://ollama.com).
2. Run `ollama pull gemma4:e2b` and `ollama pull embeddinggemma`.
3. Download the latest zip from the [Releases](../../releases) page, unzip it, and double-click `start.bat` (Windows) or run `sh start.sh`.
4. Optional, for voice recordings: double-click `add-voice.bat` once.

Check the download against the SHA-256 shown on the release page. In PowerShell: `Get-FileHash .\lamplight-offline-*.zip -Algorithm SHA256`.

## Your data
Your files stay in folders next to `server.py` (`docs`, `sessions`, `guides`, `facts.json`). `python server.py backup` zips them. Only your own computer can open the app.

## Honest limits
- Small local models can be wrong. A misheard recording or misread picture will be wrong too, so use *View text* and check important answers.
- The download is not digitally signed, so Windows may warn you.
- The Cloud login is a convenience lock on a personal PC, not a security boundary against someone who can read your files.
- Running it for several people on a server (public mode) exists but is experimental: one chat is answered at a time, and there is no email password reset.

## License
MIT. See [LICENSE](LICENSE). The AI models you download through Ollama have their own terms.
