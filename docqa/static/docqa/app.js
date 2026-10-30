(function () {
  "use strict";

  const L = window.RagLib;
  const $ = (id) => document.getElementById(id);

  const store = {
    get(key) { try { return localStorage.getItem(key); } catch (e) { return null; } },
    set(key, value) { try { localStorage.setItem(key, value); } catch (e) { /* private mode: fine */ } },
    del(key) { try { localStorage.removeItem(key); } catch (e) { /* ignore */ } },
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
    $("messages").replaceChildren($("empty"));
    $("empty").hidden = false;
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
    const list = $("doc-list");
    list.replaceChildren(...state.docs.map(docItem));
    $("doc-empty").hidden = state.docs.length > 0;

    const scope = $("scope");
    const previous = scope.value;
    scope.replaceChildren(el("option", { value: "", text: "All documents" }),
      ...state.docs.map((d) => el("option", { value: String(d.id), text: d.title })));
    scope.value = state.docs.some((d) => String(d.id) === previous) ? previous : "";

    updateComposer();
  }

  /* ---------- answer mode ---------- */

  const MODES = ["auto", "documents", "general"];
  const savedMode = store.get("mode");
  $("mode").value = MODES.includes(savedMode) ? savedMode : "auto";

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
  }

  $("mode").addEventListener("change", () => { store.set("mode", $("mode").value); updateComposer(); });

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
    return el("li", {},
      el("div", { class: "meta" },
        el("div", { class: "title", text: doc.title, title: doc.title }),
        el("div", { class: "muted", text: L.plural(doc.chunk_count, "passage") })),
      del);
  }

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
      item.querySelector(".bar").remove();
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
    xhr.upload.onload = () => { bar.style.width = "90%"; status.textContent = "Indexing..."; };
    xhr.onerror = () => fail("Upload failed: could not reach the server.");
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

  /* ---------- chat ---------- */

  const messages = $("messages");

  function nearBottom() {
    return messages.scrollHeight - messages.scrollTop - messages.clientHeight < 90;
  }
  function scrollDown() { messages.scrollTop = messages.scrollHeight; }

  function addUserMessage(text) {
    $("empty").hidden = true;
    messages.append(el("div", { class: "msg user" }, el("div", { class: "bubble", text })));
    scrollDown();
  }

  function addAssistantMessage() {
    $("empty").hidden = true;
    const typing = el("span", { class: "typing", "aria-label": "Searching your documents" }, el("i"), el("i"), el("i"));
    const answer = el("div", { class: "answer" }, typing);
    const notice = el("div", { class: "notice", hidden: "" });
    const badge = el("div", { class: "badge", hidden: "" });
    const searched = el("div", { class: "muted searched", hidden: "" });
    const sourcesBox = el("div");
    const errorBox = el("div");
    const bubble = el("div", { class: "bubble" }, badge, searched, notice, answer, errorBox, sourcesBox);
    messages.append(el("div", { class: "msg assistant" }, bubble));
    scrollDown();
    return { bubble, answer, notice, badge, searched, sourcesBox, errorBox };
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

  function renderSources(view, sources) {
    if (!sources.length) return view.sourcesBox.replaceChildren();
    const cards = sources.map((s) => el("li", { class: "source", "data-n": String(s.n) },
      el("header", {}, "[" + s.n + "] " + s.document_title + " ",
        el("span", { text: "· passage " + (s.chunk_index + 1) + " · similarity " + s.score.toFixed(2) })),
      el("p", { text: s.text })));
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

  async function ask(question) {
    addUserMessage(question);
    const view = addAssistantMessage();
    setAsking(true);
    state.controller = new AbortController();

    let sources = [];
    let meta = null;
    let hadError = false;
    let text = "";
    let frame = 0;
    const paint = () => {
      frame = 0;
      const stick = nearBottom();
      const shown = L.stripFallback(text);
      view.notice.hidden = !shown.fallback;
      if (shown.fallback) view.notice.textContent = "No language model is configured on the server, so this is the best matching passage rather than a written answer.";
      view.answer.innerHTML = shown.text ? L.renderAnswer(shown.text, sources.length) : "";
      if (stick) scrollDown();
    };
    const schedule = () => { if (!frame) frame = requestAnimationFrame(paint); };
    const showError = (message) => {
      if (!text) view.answer.replaceChildren();
      view.errorBox.replaceChildren(el("p", { class: "error", role: "alert", text: message }));
    };

    try {
      const mode = $("mode").value;
      const body = { question, mode, history: L.buildHistory(state.chat), top_k: Number($("topk").value) };
      if (mode !== "general" && $("scope").value) body.document_id = Number($("scope").value);
      const resp = await api("/api/ask/stream/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: state.controller.signal,
      });
      if (!resp.ok) { showError(await readError(resp)); return; }

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
          if (event === "sources") { sources = data.sources; meta = data; renderSources(view, sources); showOrigin(view, data); }
          else if (event === "token") { text += data.t; schedule(); }
          else if (event === "error") { finished = hadError = true; paint(); showError(data.detail); }
          else if (event === "done") { finished = true; }
        }
      }
      paint();
      if (!finished) { hadError = true; showError("The connection closed before the answer finished."); }
      remember(question, text, meta, hadError);
    } catch (err) {
      paint();
      if (err.name === "AbortError") view.errorBox.replaceChildren(el("p", { class: "muted", text: "Stopped." }));
      else showError("Could not reach the server. Is it running?");
    } finally {
      if (frame) cancelAnimationFrame(frame);
      paint();
      state.controller = null;
      setAsking(false);
    }
  }

  /** Keep a finished exchange as memory. Failed, stopped, "not found" and passage-only replies are left out. */
  function remember(question, text, meta, hadError) {
    if (hadError || !meta || meta.mode === "not_found" || !text.trim()) return;
    const answer = L.stripFallback(text);
    if (answer.fallback) return;
    state.chat.push({ role: "user", content: question }, { role: "assistant", content: answer.text });
  }

  $("new-chat").addEventListener("click", () => {
    if (state.controller) state.controller.abort();
    state.chat = [];
    messages.replaceChildren($("empty"));
    $("empty").hidden = false;
    $("input").focus();
  });

  $("composer").addEventListener("submit", (event) => {
    event.preventDefault();
    const question = $("input").value.trim();
    if (!question || state.asking || !canAsk()) return;
    $("input").value = "";
    autoGrow();
    ask(question);
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
    try {
      await refreshDocs();
    } catch (err) {
      messages.prepend(el("p", { class: "error", text: "Could not reach the server." }));
    }
    if (state.token) $("input").focus();
  }

  setMode("signin");
  boot();
})();
