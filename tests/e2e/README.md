# Browser end-to-end checks

The Python and JavaScript test suites cover logic. These checks cover what only a real browser shows:
streaming, focus, scrolling, layout. They run against the real app and the real neural embedder, with the
fake model server from `tests/fake_llm.py`, so no key, quota or network is needed.

```bash
python tests/e2e/server.py          # http://127.0.0.1:8010, throwaway database
```

Change which fake model the app talks to at any time, without restarting, by writing
`primary|fallback1,fallback2` to `tests/e2e/model.txt` (ignored by git). Model names are explained at the
top of `tests/fake_llm.py`: `good`, `busy`, `denied`, `cut`, `slow`, `decline`, `markdown`, `special`...

## Checklist (all passed on the last full run)

Sign in and documents
- [ ] Wrong password shows "Unable to log in"; a weak password shows the server's reasons; a valid one signs in
- [ ] Reload keeps you signed in; a stale token returns you to the sign-in page with a message
- [ ] Drag-and-drop or choose files: good files index, an empty file / `.exe` / file over 5 MB are refused with a reason
- [ ] Documents list shows passages and age ("just now"); delete asks "Sure?" and reverts after 3 s

Answers
- [ ] Suggestion chips appear with documents present and send the question when clicked
- [ ] "Summarize this document" reads passages spread across the documents, labelled "sampled from the document"
- [ ] Status line goes "Searching..." then "Writing an answer from N passages..."
- [ ] Footer shows the time (`0.2 s · 4 passages`) and a Copy button
- [ ] Source cards highlight the question's words; clicking a `[1]` opens and highlights that source
- [ ] Model `markdown`: heading, bullet and numbered lists, bold, inline code, a code block render; raw HTML shows as text
- [ ] Model `special`: newlines, quotes, backslashes, emoji, CJK and U+2028 arrive intact

Modes and memory
- [ ] Auto: a relevant question is "From your documents"; an unrelated one is "General knowledge" with no sources
- [ ] Model `decline`: the badge switches to general knowledge, sources vanish, status says "Not covered by your documents"
- [ ] Documents only refuses an unrelated question without calling the model; General chat ignores the documents
- [ ] A follow-up shows "Searched for: ..."; failed, stopped and "not found" replies are not remembered
- [ ] New chat clears the conversation and the memory

Failures and controls
- [ ] Model `denied`: a clear error, the sources stay visible, "Try again" replaces the failed exchange and succeeds once the model is back
- [ ] Model `slow`: Stop (and the Esc key) ends the stream, no late text arrives, "Try again" is offered
- [ ] `/` and Ctrl+K focus the question box, but `/` is ignored while typing in a dropdown
- [ ] After many messages, "Latest" appears when scrolled up and jumps back down

Layout and theme
- [ ] Phone width (375 px): no horizontal scrolling, the top bar fits on one line, the document drawer opens and closes
- [ ] Light and dark themes both readable; the browser console shows no errors
