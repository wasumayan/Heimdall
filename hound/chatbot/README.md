# Hound Chatbot (Realtime WebRTC)

A minimal voice + text chat interface for Hound using the Realtime API. Includes function calling stubs and a simple portrait avatar that switches based on emotion/state.

## Features
- WebRTC audio out (model voice), mic input (push-to-talk or VAD-friendly)
- Text chat via data channel
- Function-calling stub pipeline: model -> function_call -> backend endpoint -> function_call_output -> model reply
- Minimal avatar: four portraits (normal, concentrated, amused, shy) driven by transcript/heuristics

## Run

1) Source keys and install deps

```
source ../../API_KEYS.txt
pip install -r ../../custom/requirements.txt
```

2) Start server

```
python hound/chatbot/run.py
```

3) Open UI

```
http://127.0.0.1:5280
```

## Notes
- Replace placeholder portraits in `hound/chatbot/static/avatars/` with your own art.
- The UI can set an active project (in the Audit Stream box). The server stores it and tools use it by default. You can also pass `project_id` explicitly in tool calls.
- Tools exposed under `/api/tool/*`:
  - `get_hound_status` — reads the latest session file for the active project and returns status, coverage, and token usage totals.
  - `enqueue_steering` — appends a line to `<project>/.hound/steering.jsonl` for agent consumption.
