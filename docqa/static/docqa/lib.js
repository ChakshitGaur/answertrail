/* Pure helpers for the chat UI. Works in the browser (window.RagLib) and in Node (tests). */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.RagLib = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const FALLBACK_PREFIX = "[extractive fallback, no LLM configured] ";
  const ALLOWED_EXTENSIONS = [".pdf", ".txt", ".md"];
  const MAX_UPLOAD_BYTES = 5 * 1024 * 1024;

  const ESCAPES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };

  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (c) => ESCAPES[c]);
  }

  /** Splits the server's "no model configured" prefix off the answer text. */
  function stripFallback(text) {
    if (text.startsWith(FALLBACK_PREFIX)) return { text: text.slice(FALLBACK_PREFIX.length), fallback: true };
    return { text, fallback: false };
  }

  /**
   * Turns model/document text into safe HTML. Everything is escaped first, so nothing in an
   * uploaded document or a model reply can inject markup; only then are our own tags added.
   * Citations like [1] or [1, 2] become buttons, but only for sources that exist.
   */
  function renderAnswer(text, sourceCount) {
    const cite = (n) => `<button type="button" class="cite" data-n="${n}" title="Show source ${n}">${n}</button>`;
    let html = escapeHtml(text);
    html = html.replace(/`([^`\n]+)`/g, "<code>$1</code>");
    html = html.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/\[(\d{1,2}(?:\s*,\s*\d{1,2})*)\]/g, (match, list) => {
      const numbers = list.split(",").map((s) => Number(s.trim()));
      if (!numbers.every((n) => n >= 1 && n <= sourceCount)) return match;
      return numbers.map(cite).join("");
    });
    return html
      .split(/\n{2,}/)
      .map((p) => `<p>${p.replace(/\n/g, "<br>")}</p>`)
      .join("");
  }

  /**
   * Incremental Server-Sent-Events parser. Feed it everything received so far; it returns the
   * complete events plus the unfinished remainder to prepend to the next chunk.
   */
  function parseSSE(buffer) {
    const parts = buffer.replace(/\r\n/g, "\n").split("\n\n");
    const rest = parts.pop();
    const events = [];
    for (const block of parts) {
      let event = "message";
      const data = [];
      for (const line of block.split("\n")) {
        if (line.startsWith(":")) continue; // comment / keep-alive
        const i = line.indexOf(":");
        const field = i === -1 ? line : line.slice(0, i);
        const value = i === -1 ? "" : line.slice(i + 1).replace(/^ /, "");
        if (field === "event") event = value;
        else if (field === "data") data.push(value);
      }
      if (!data.length) continue;
      try {
        events.push({ event, data: JSON.parse(data.join("\n")) });
      } catch (err) {
        events.push({ event: "error", data: { detail: "The server sent a malformed event." } });
      }
    }
    return { events, rest };
  }

  /** Readable message from a Django REST Framework error body (or a plain-text one). */
  function apiErrorMessage(status, body) {
    if (body && typeof body === "object" && !Array.isArray(body)) {
      if (typeof body.detail === "string") return body.detail;
      const parts = Object.entries(body).map(([field, msg]) => {
        const text = Array.isArray(msg) ? msg.join(" ") : String(msg);
        return field === "non_field_errors" ? text : `${field}: ${text}`;
      });
      if (parts.length) return parts.join(" ");
    }
    if (Array.isArray(body) && body.length) return body.join(" ");
    if (typeof body === "string" && body.trim()) return body.trim().slice(0, 200);
    return `Request failed (HTTP ${status}).`;
  }

  /** Returns an error string if the file shouldn't be uploaded, otherwise null. */
  function validateUpload(name, size) {
    const lower = name.toLowerCase();
    if (!ALLOWED_EXTENSIONS.some((ext) => lower.endsWith(ext))) return "Only PDF, TXT and MD files are supported.";
    if (size === 0) return "The file is empty.";
    if (size > MAX_UPLOAD_BYTES) return `The file is larger than ${formatBytes(MAX_UPLOAD_BYTES)}.`;
    return null;
  }

  const HISTORY_MAX_MESSAGES = 8; // keep in step with the server (docqa/services/llm.py)
  const HISTORY_MAX_CHARS = 1500;

  /** The recent conversation to send with a question, so follow-ups like "and the second one?" work. */
  function buildHistory(chat, maxMessages = HISTORY_MAX_MESSAGES) {
    return chat
      .filter((m) => m && (m.role === "user" || m.role === "assistant") && typeof m.content === "string" && m.content.trim())
      .slice(-maxMessages)
      .map((m) => ({ role: m.role, content: m.content.trim().slice(0, HISTORY_MAX_CHARS) }));
  }

  const BADGES = {
    documents: { kind: "documents", label: "From your documents" },
    general: { kind: "general", label: "General knowledge, not from your files" },
    not_found: { kind: "not_found", label: "Not in your documents" },
  };

  /** Label shown on an answer so you can tell a cited answer from a general-knowledge one. */
  function modeBadge(mode) {
    return BADGES[mode] || null;
  }

  const MODE_HELP = {
    auto: "Uses your documents when they cover the question, otherwise answers from general knowledge and says so.",
    documents: "Answers only from your documents, and says when they don't cover it.",
    general: "Ignores your documents and answers like a normal assistant.",
  };

  function plural(count, word) {
    return `${count} ${word}${count === 1 ? "" : "s"}`;
  }

  function formatBytes(n) {
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(n < 10240 ? 1 : 0)} KB`;
    return `${(n / 1024 / 1024).toFixed(1)} MB`;
  }

  return { FALLBACK_PREFIX, MAX_UPLOAD_BYTES, escapeHtml, stripFallback, renderAnswer, parseSSE, apiErrorMessage, validateUpload, formatBytes, plural, buildHistory, modeBadge, MODE_HELP, HISTORY_MAX_MESSAGES };
});
