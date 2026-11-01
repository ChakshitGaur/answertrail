(function () {
  "use strict";

  const L = window.RagLib;
  const $ = (id) => document.getElementById(id);

  const store = {
    get(key) { try { return localStorage.getItem(key); } catch (e) { return null; } },
    set(key, value) { try { localStorage.setItem(key, value); } catch (e) { /* private mode: fine */ } },
    del(key) { try { localStorage.removeItem(key); } catch (e) { /* ignore */ }  },
  };

  const state = {
    token: store.get("token"),
    username: store.get("username") || "",
    docs: [],
    chat: [], // finished question/answer pairs sent along as memory, so follow-ups work
    asking: false,
    controller: null,
    mode: "signin",
  };

  /* ---------- helpers ---------- */

  function el(tag, props, ...children) {
    const node = document.createElement(tag);
    Object.entries(props || {}).forEach(([k, v]) => {
      if (k === "class") node.className = v;
      else if (k === "text") node.textContent = v;
      else node.setAttribute(k, v);
    });
    children.forEach((c) => c && node.append(c));
    return node;
  }

  async function readError(resp) {
    let body;
    try { body = await resp.clone().json(); } catch (e) { try { body = await resp.text(); } catch (e2) { body = null; } }
    return L.apiErrorMessage(resp.status, body);
  }

  async function api(path, options) {
    const opts = Object.assign({}, options);
    opts.headers = Object.assign({}, opts.headers);
    if (state.token) opts.headers.Authorization = "Token " + state.token;
    const resp = await fetch(path, opts);
    if (resp.status === 401 && state.token) signOut("Your session expired. Please sign in again.");
    return resp;
  }

  async function copyText(text) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch (e) {
      const box = el("textarea", { "aria-hidden": "true" });
      box.value = text;
      box.style.position = "fixed";
      box.style.opacity = "0";
      document.body.append(box);
      box.select();
      let ok = false;
      try { ok = document.execCommand("copy"); } catch (e2) { /* not allowed here */ }
      box.remove();
      return ok;
    }
  }

  /* ---------- theme ---------- */

  function applyTheme(theme) {
    if (theme === "light" || theme === "dark") document.documentElement.dataset.theme = theme;
    else delete document.documentElement.dataset.theme;
  }
  applyTheme(store.get("theme"));
  $("theme-btn").addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    const next = dark ? "light" : "dark";
    store.set("theme", next);
    applyTheme(next);
  });

  /* ---------- auth ---------- */

  function setMode(mode) {
    state.mode = mode;
    $("tab-signin").setAttribute("aria-selected", String(mode === "signin"));
    $("tab-register").setAttribute("aria-selected", String(mode === "register"));
    $("auth-submit").textContent = mode === "signin" ? "Sign in" : "Create account";
    $("password").autocomplete = mode === "signin" ? "current-password" : "new-password";
    $("auth-error").hidden = true;
  }
  $("tab-signin").addEventListener("click", () => setMode("signin"));
  $("tab-register").addEventListener("click", () => setMode("register"));

  function authError(message) {
    $("auth-error").textContent = message;
    $("auth-error").hidden = false;
  }

  $("auth-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const username = $("username").value.trim();
    const password = $("password").value;
    if (!username || !password) return authError("Enter a username and password.");
    $("auth-submit").disabled = true;
    try {
      const path = state.mode === "signin" ? "/api/auth/token/" : "/api/auth/register/";
      const resp = await fetch(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });
      if (!resp.ok) return authError(await readError(resp));
      state.token = (await resp.json()).token;
      state.username = username;
      store.set("token", state.token);
      store.set("username", username);
      $("password").value = "";
      await boot();
    } catch (err) {
      authError("Could not reach the server. Is it running?");
    } finally {
      $("auth-submit").disabled = false;
    }
  });

  function signOut(message) {
    if (state.controller) state.controller.abort();
    state.token = null;
    state.docs = [];
    state.chat = [];
    store.del("token");
    store.del("username");
    messages.replaceChildren($("empty"));
    $("empty").hidden = false;
    updateJump();
    showAuth(message);
  }
  $("logout-btn").addEventListener("click", () => signOut());

  function showAuth(message) {
    $("app").hidden = true;
    $("auth").hidden = false;
    if (message) authError(message); else $("auth-error").hidden = true;
    $("username").focus();
  }

  /* ---------- documents ---------- */

  async function refreshDocs() {
    const resp = await api("/api/documents/");
    if (!resp.ok) return;
    state.docs = await resp.json();
    renderDocs();
  }

  function renderDocs() {
    $("doc-list").replaceChildren(...state.docs.map(docItem));
    $("doc-empty").textContent = "Nothing uploaded yet.";
    $("doc-empty").hidden = state.docs.length > 0;

    const scope = $("scope");
    const previous = scope.value;
    scope.replaceChildren(el("option", { value: "", text: "All documents" }),
      ...state.docs.map((d) => el("option", { value: String(d.id), text: d.title })));
    scope.value = state.docs.some((d) => String(d.id) === previous) ? previous : "";

    updateComposer();
  }

  function docItem(doc) {
    const del = el("button", { type: "button", text: "Delete", "aria-label": "Delete " + doc.title });
    let timer = null;
    del.addEventListener("click", async () => {
      if (!del.classList.contains("confirm")) {
        del.classList.add("confirm");
        del.textContent = "Sure?";
        timer = setTimeout(() => { del.classList.remove("confirm"); del.textContent = "Delete"; }, 3000);
        return;
      }
      clearTimeout(timer);
      del.disabled = true;
      const resp = await api("/api/documents/" + doc.id + "/", { method: "DELETE" });
      if (resp.ok || resp.status === 404) await refreshDocs();
      else { del.disabled = false; del.textContent = "Failed"; }
    });
    const when = L.timeAgo(doc.created_at);
    return el("li", {},
      el("div", { class: "meta" },
        el("div", { class: "title", text: doc.title, title: doc.title }),
        el("div", { class: "muted", text: L.plural(doc.chunk_count, "passage") + (when ? " · " + when : "") })),
      del);
  }

  /* ---------- answer mode and composer ---------- */

  const MODES = ["auto", "documents", "general"];
  const savedMode = store.get("mode");
  $("mode").value = MODES.includes(savedMode) ? savedMode : "auto";
  if ([...$("topk").options].some((o) => o.value === store.get("topk"))) $("topk").value = store.get("topk");

  // Documents-only needs a document; Auto and General chat work without one.
  function canAsk() {
    return $("mode").value !== "documents" || state.docs.length > 0;
  }

  function updateComposer() {
    const mode = $("mode").value;
    const hasDocs = state.docs.length > 0;
    const ok = canAsk();
    $("input").disabled = !ok || state.asking;
    $("send").disabled = !ok || state.asking;
    $("input").placeholder = !ok ? "Upload a document to start" : mode === "general" ? "Ask anything" : "Ask about your documents";
    $("mode-help").textContent = L.MODE_HELP[mode];
    $("scope").disabled = mode === "general";
    $("topk").disabled = mode === "general";
    $("empty-text").textContent =
      mode === "general" ? "General chat: ask anything. Your documents are not used in this mode."
      : !hasDocs && mode === "documents" ? "Upload a document on the left, then ask a question."
      : mode === "auto" ? "Ask anything. If your documents cover it, the answer cites them; otherwise you get a general answer, labelled as such."
      : "Ask anything about your documents. Answers cite the passages they came from.";
    renderSuggestions(hasDocs && mode !== "general");
  }

  function renderSuggestions(show) {
    const box = $("suggestions");
    box.hidden = !show;
    if (!show || box.childElementCount) return;
    L.SUGGESTIONS.forEach((text) => {
      const chip = el("button", { type: "button", text });
      chip.addEventListener("click", () => submitQuestion(text));
      box.append(chip);
    });
  }

  $("mode").addEventListener("change", () => { store.set("mode", $("mode").value); updateComposer(); });
  $("topk").addEventListener("change", () => store.set("topk", $("topk").value));

  /* ---------- uploads ---------- */

  const dropzone = $("dropzone");
  dropzone.addEventListener("click", () => $("file-input").click());
  dropzone.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("file-input").click(); } });
  ["dragenter", "dragover"].forEach((t) => dropzone.addEventListener(t, (e) => { e.preventDefault(); dropzone.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => dropzone.addEventListener(t, (e) => { e.preventDefault(); dropzone.classList.remove("over"); }));
  dropzone.addEventListener("drop", (e) => uploadFiles(e.dataTransfer.files));
  $("file-input").addEventListener("change", (e) => { uploadFiles(e.target.files); e.target.value = ""; });

  function uploadFiles(files) {
    Array.from(files).forEach(uploadFile);
  }

  function uploadFile(file) {
    const status = el("div", { class: "muted", text: "Uploading..." });
    const bar = el("span");
    const item = el("li", {}, el("div", { text: file.name }), status, el("div", { class: "bar" }, bar));
    $("uploads").prepend(item);

    const fail = (message) => {
      status.textContent = message;
      status.className = "failed";
      const barBox = item.querySelector(".bar");
      if (barBox) barBox.remove();
      setTimeout(() => item.remove(), 10000);
    };
    const problem = L.validateUpload(file.name, file.size);
    if (problem) return fail(problem);

    const form = new FormData();
    form.append("file", file);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/documents/");
    xhr.setRequestHeader("Authorization", "Token " + state.token);
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) bar.style.width = Math.round((e.loaded / e.total) * 90) + "%"; };
    xhr.upload.onload = () => { bar.style.width = "90%"; status.textContent = "Indexing... large files can take a minute"; };
    xhr.onerror = () => fail("Upload failed: could not reach the server.");
    xhr.ontimeout = () => fail("Upload timed out. Try a smaller file.");
    xhr.onload = async () => {
      if (xhr.status === 401) return signOut("Your session expired. Please sign in again.");
      let body = null;
      try { body = JSON.parse(xhr.responseText); } catch (e) { /* not JSON */ }
      if (xhr.status === 201) {
        bar.style.width = "100%";
        status.textContent = "Done: " + L.plural(body.chunk_count, "passage");
        setTimeout(() => item.remove(), 4000);
        await refreshDocs();
      } else {
        fail(L.apiErrorMessage(xhr.status, body));
      }
    };
    xhr.send(form);
  }

  /* ---------- chat view ---------- */

  const messages = $("messages");

  function nearBottom() {
    return messages.scrollHeight - messages.scrollTop - messages.clientHeight < 90;
  }
  function scrollDown() { messages.scrollTop = messages.scrollHeight; updateJump(); }
  function updateJump() { $("jump").hidden = nearBottom() || !messages.querySelector(".msg"); }
  let jumpFrame = 0;
  messages.addEventListener("scroll", () => {
    if (!jumpFrame) jumpFrame = requestAnimationFrame(() => { jumpFrame = 0; updateJump(); });
  });
  $("jump").addEventListener("click", scrollDown);

  function addUserMessage(text) {
    $("empty").hidden = true;
    const node = el("div", { class: "msg user" }, el("div", { class: "bubble", text }));
    messages.append(node);
    scrollDown();
    return node;
  }

  function addAssistantMessage(waitingText) {
    $("empty").hidden = true;
    const status = el("div", { class: "status", "aria-live": "polite", text: waitingText });
    const typing = el("span", { class: "typing", "aria-hidden": "true" }, el("i"), el("i"), el("i"));
    const answer = el("div", { class: "answer" }, typing);
    const notice = el("div", { class: "notice", hidden: "" });
    const badge = el("div", { class: "badge", hidden: "" });
    const searched = el("div", { class: "muted searched", hidden: "" });
    const sourcesBox = el("div");
    const errorBox = el("div");
    const tools = el("div", { class: "tools", hidden: "" });
    const bubble = el("div", { class: "bubble" }, badge, searched, status, notice, answer, errorBox, sourcesBox, tools);
    const node = el("div", { class: "msg assistant" }, bubble);
    messages.append(node);
    scrollDown();
    return { node, bubble, status, answer, notice, badge, searched, sourcesBox, errorBox, tools, rawText: "" };
  }

  /** Tells the reader where an answer comes from, and what was actually searched for follow-ups. */
  function showOrigin(view, meta) {
    const badge = L.modeBadge(meta.mode);
    if (badge) {
      view.badge.className = "badge " + badge.kind;
      view.badge.textContent = badge.label;
      view.badge.hidden = false;
    }
    if (meta.searched_for) {
      view.searched.textContent = "Searched for: " + meta.searched_for;
      view.searched.hidden = false;
    }
  }

  function renderSources(view, sources, query) {
    if (!sources.length) return view.sourcesBox.replaceChildren();
    const cards = sources.map((s) => {
      const passage = el("p", {});
      passage.innerHTML = L.highlight(s.text, query); // escaped by highlight(); only <mark> is added
      return el("li", { class: "source", "data-n": String(s.n) },
        el("header", {}, "[" + s.n + "] " + s.document_title + " ",
          el("span", { text: "· passage " + (s.chunk_index + 1) + " · " + (typeof s.score === "number" ? "similarity " + s.score.toFixed(2) : "sampled from the document") })),
        passage);
    });
    view.sourcesBox.replaceChildren(el("details", { class: "sources" },
      el("summary", { text: "Sources (" + sources.length + ")" }), el("ol", {}, ...cards)));
  }

  messages.addEventListener("click", (e) => {
    const cite = e.target.closest(".cite");
    if (!cite) return;
    const bubble = cite.closest(".bubble");
    const details = bubble.querySelector("details.sources");
    if (!details) return;
    details.open = true;
    bubble.querySelectorAll(".source.hl").forEach((n) => n.classList.remove("hl"));
    const card = details.querySelector('.source[data-n="' + cite.dataset.n + '"]');
    if (card) { card.classList.add("hl"); card.scrollIntoView({ block: "nearest", behavior: "smooth" }); }
  });

  function setAsking(on) {
    state.asking = on;
    $("send").hidden = on;
    $("stop").hidden = !on;
    updateComposer();
    if (!on) $("input").focus();
  }

  /** Copy / try-again controls and the timing line under a finished answer. */
  function showTools(view, { summary, retry }) {
    view.tools.replaceChildren();
    if (summary) view.tools.append(el("span", { class: "summary", text: summary }));
    else view.tools.append(el("span", { class: "summary" }));
    if (view.rawText.trim()) {
      const copy = el("button", { type: "button", text: "Copy" });
      copy.addEventListener("click", async () => {
        copy.textContent = (await copyText(view.rawText)) ? "Copied" : "Copy failed";
        setTimeout(() => { copy.textContent = "Copy"; }, 1500);
      });
      view.tools.append(copy);
    }
    if (retry) {
      const again = el("button", { type: "button", text: "Try again" });
      again.addEventListener("click", retry);
      view.tools.append(again);
    }
    view.tools.hidden = false;
  }

  async function ask(question) {
    const mode = $("mode").value;
    const userNode = addUserMessage(question);
    const view = addAssistantMessage(mode === "general" ? "Thinking..." : "Searching your documents...");
    setAsking(true);
    state.controller = new AbortController();

    const startedAt = performance.now();
    let firstWordAt = null;
    let sources = [];
    let meta = null;
    let hadError = false;
    let text = "";
    let frame = 0;

    const tick = setInterval(() => {
      const seconds = Math.round((performance.now() - startedAt) / 1000);
      if (!view.status.hidden && seconds >= 8) view.status.textContent = "Still working... " + seconds + " s (the model may be busy)";
    }, 1000);

    const paint = () => {
      frame = 0;
      const stick = nearBottom();
      const shown = L.stripFallback(text);
      view.rawText = shown.text;
      view.notice.hidden = !shown.fallback;
      if (shown.fallback) view.notice.textContent = "No language model is configured on the server, so this is the best matching passage rather than a written answer.";
      if (shown.text) {
        if (firstWordAt === null) { firstWordAt = performance.now() - startedAt; view.status.hidden = true; }
        view.answer.innerHTML = L.renderAnswer(shown.text, sources.length);
      } else {
        view.answer.innerHTML = "";
      }
      if (stick) scrollDown();
    };
    const schedule = () => { if (!frame) frame = requestAnimationFrame(paint); };
    const showError = (message) => {
      view.status.hidden = true;
      if (!text) view.answer.replaceChildren();
      view.errorBox.replaceChildren(el("p", { class: "error", role: "alert", text: message }));
    };
    const retry = () => { userNode.remove(); view.node.remove(); updateJump(); submitQuestion(question); };

    try {
      const body = { question, mode, history: L.buildHistory(state.chat), top_k: Number($("topk").value) };
      if (mode !== "general" && $("scope").value) body.document_id = Number($("scope").value);
      const resp = await api("/api/ask/stream/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: state.controller.signal,
      });
      if (!resp.ok) { hadError = true; showError(await readError(resp)); return; }

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let finished = false;
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parsed = L.parseSSE(buffer);
        buffer = parsed.rest;
        for (const { event, data } of parsed.events) {
          if (event === "sources") {
            sources = data.sources;
            meta = data;
            renderSources(view, sources, data.searched_for || question);
            showOrigin(view, data);
            view.status.textContent = data.overview ? "Reading " + L.plural(sources.length, "passage") + " spread across your documents..."
              : data.mode === "documents" ? "Writing an answer from " + L.plural(sources.length, "passage") + "..."
              : data.mode === "general" ? "Answering from general knowledge..." : "";
            if (data.mode === "not_found") view.status.hidden = true;
          } else if (event === "mode") {
            // the model said the passages don't answer it: relabel, drop the sources, answer generally
            meta = { ...meta, mode: data.mode };
            sources = data.sources;
            text = "";
            renderSources(view, sources, question);
            showOrigin(view, meta);
            view.status.hidden = false;
            view.status.textContent = "Not covered by your documents. Answering from general knowledge...";
            schedule();
          } else if (event === "token") {
            text += data.t;
            schedule();
          } else if (event === "error") {
            finished = hadError = true;
            paint();
            showError(data.detail);
          } else if (event === "done") {
            finished = true;
          }
        }
      }
      paint();
      if (!finished) { hadError = true; showError("The connection closed before the answer finished."); }
      remember(question, text, meta, hadError);
    } catch (err) {
      paint();
      if (err.name === "AbortError") {
        view.status.hidden = true;
        view.errorBox.replaceChildren(el("p", { class: "muted", text: "Stopped." }));
        hadError = true;
      } else {
        hadError = true;
        showError("Could not reach the server. Is it running?");
      }
    } finally {
      clearInterval(tick);
      if (frame) cancelAnimationFrame(frame);
      paint();
      view.status.hidden = true;
      const total = performance.now() - startedAt;
      if (meta && !hadError) meta = { ...meta, sources: sources.length };
      showTools(view, {
        summary: hadError ? "" : L.answerSummary(meta, total, firstWordAt),
        retry: hadError ? retry : null,
      });
      state.controller = null;
      setAsking(false);
      scrollDown();
    }
  }

  /** Keep a finished exchange as memory. Failed, stopped, "not found" and passage-only replies are left out. */
  function remember(question, text, meta, hadError) {
    if (hadError || !meta || meta.mode === "not_found" || !text.trim()) return;
    const answer = L.stripFallback(text);
    if (answer.fallback) return;
    state.chat.push({ role: "user", content: question }, { role: "assistant", content: answer.text });
  }

  function submitQuestion(question) {
    question = question.trim();
    if (!question || state.asking || !canAsk()) return;
    ask(question);
  }

  $("new-chat").addEventListener("click", () => {
    if (state.controller) state.controller.abort();
    state.chat = [];
    messages.replaceChildren($("empty"));
    $("empty").hidden = false;
    updateJump();
    $("input").focus();
  });

  $("composer").addEventListener("submit", (event) => {
    event.preventDefault();
    const question = $("input").value.trim();
    if (!question || state.asking || !canAsk()) return;
    $("input").value = "";
    autoGrow();
    submitQuestion(question);
  });
  $("stop").addEventListener("click", () => state.controller && state.controller.abort());
  $("input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); $("composer").requestSubmit(); }
  });
  function autoGrow() {
    const box = $("input");
    box.style.height = "auto";
    const borders = box.offsetHeight - box.clientHeight; // border-box: scrollHeight leaves them out
    box.style.height = Math.min(box.scrollHeight + borders, 160) + "px";
    box.style.overflowY = box.scrollHeight + borders > 160 ? "auto" : "hidden";
  }
  $("input").addEventListener("input", autoGrow);

  /* ---------- keyboard shortcuts ---------- */

  document.addEventListener("keydown", (e) => {
    if ($("app").hidden) return;
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName);
    if (e.key === "Escape" && state.controller) state.controller.abort();
    else if ((e.key === "/" && !typing) || ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k")) {
      e.preventDefault();
      if (!$("input").disabled) $("input").focus();
    }
  });

  /* ---------- mobile drawer ---------- */

  function toggleDrawer(open) {
    $("sidebar").classList.toggle("open", open);
    $("scrim").hidden = !open;
  }
  $("menu-btn").addEventListener("click", () => toggleDrawer(!$("sidebar").classList.contains("open")));
  $("scrim").addEventListener("click", () => toggleDrawer(false));

  /* ---------- start ---------- */

  async function boot() {
    if (!state.token) return showAuth();
    $("who").textContent = state.username;
    $("auth").hidden = true;
    $("app").hidden = false;
    $("doc-empty").textContent = "Loading your documents...";
    $("doc-empty").hidden = false;
    try {
      await refreshDocs();
    } catch (err) {
      $("doc-empty").textContent = "Could not reach the server.";
    }
    if (state.token) $("input").focus();
  }

  setMode("signin");
  boot();
})();
