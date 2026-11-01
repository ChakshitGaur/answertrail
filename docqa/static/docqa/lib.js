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

  /* ---------- answer rendering ---------- */

  const LIST_ITEM = /^\s*(?:([-*•])|(\d{1,3})[.)])\s+(.+)$/;
  const HEADING = /^\s{0,3}#{1,4}\s+(.+)$/;
  const FENCE = /^\s*```/;

  /** Inline formatting on text that is ALREADY escaped: `code`, **bold** and [n] citation buttons. */
  function inline(escaped, sourceCount) {
    const cite = (n) => `<button type="button" class="cite" data-n="${n}" title="Show source ${n}">${n}</button>`;
    return escaped
      .split(/(`[^`\n]+`)/) // odd entries are code spans, whose contents are left exactly as they are
      .map((part, i) => {
        if (i % 2) return `<code>${part.slice(1, -1)}</code>`;
        return part
          .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
          .replace(/\[(\d{1,2}(?:\s*,\s*\d{1,2})*)\]/g, (match, list) => {
            const numbers = list.split(",").map((s) => Number(s.trim()));
            if (!numbers.every((n) => n >= 1 && n <= sourceCount)) return match;
            return numbers.map(cite).join("");
          });
      })
      .join("");
  }

  /**
   * Turns model/document text into safe HTML: paragraphs, bullet and numbered lists, headings, fenced
   * code blocks, `code`, **bold** and citation buttons. Everything is escaped first, so nothing in an
   * uploaded document or a model reply can inject markup; only then are our own tags added. Citations
   * like [1] or [1, 2] become buttons only for sources that exist, and never inside code.
   */
  function renderAnswer(text, sourceCount) {
    const lines = escapeHtml(text).replace(/\r\n?/g, "\n").split("\n");
    const out = [];
    let paragraph = [];
    const flush = () => {
      if (paragraph.length) out.push(`<p>${paragraph.map((l) => inline(l, sourceCount)).join("<br>")}</p>`);
      paragraph = [];
    };

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      if (FENCE.test(line)) {
        flush();
        const code = [];
        i++;
        while (i < lines.length && !FENCE.test(lines[i])) code.push(lines[i++]);
        out.push(`<pre><code>${code.join("\n")}</code></pre>`); // an unclosed fence (still streaming) runs to the end
        continue;
      }
      const item = LIST_ITEM.exec(line);
      if (item) {
        flush();
        const ordered = item[2] !== undefined;
        const items = [];
        let m;
        while (i < lines.length && (m = LIST_ITEM.exec(lines[i])) && (m[2] !== undefined) === ordered) {
          items.push(`<li>${inline(m[3], sourceCount)}</li>`);
          i++;
        }
        i--;
        out.push(ordered ? `<ol>${items.join("")}</ol>` : `<ul>${items.join("")}</ul>`);
        continue;
      }
      const heading = HEADING.exec(line);
      if (heading) {
        flush();
        out.push(`<h4>${inline(heading[1], sourceCount)}</h4>`);
        continue;
      }
      if (!line.trim()) {
        flush();
        continue;
      }
      paragraph.push(line);
    }
    flush();
    return out.join("");
  }

  const QUESTION_STOPWORDS = new Set(
    ("the and for are was were what which who whom when where how why does did can could would should this that " +
      "with from have has had you your about tell explain simply please give show").split(" ")
  );

  /** The words of a question worth highlighting in a source passage. */
  function queryTerms(query) {
    const words = (String(query).toLowerCase().match(/[a-z0-9]{3,}/g) || []).filter((w) => !QUESTION_STOPWORDS.has(w));
    return Array.from(new Set(words)).slice(0, 8);
  }

  /** Escaped passage text with the question's words wrapped in <mark> (whole words, plural allowed). */
  function highlight(text, query) {
    const html = escapeHtml(text);
    const terms = queryTerms(query);
    if (!terms.length) return html;
    // terms are plain letters/digits, so they are safe in a pattern; entities like &amp; are skipped whole
    const pattern = new RegExp(`(&(?:[a-z]+|#\\d+);)|((?<![a-z0-9])(?:${terms.join("|")})s?(?![a-z0-9]))`, "gi");
    return html.replace(pattern, (match, entity, word) => (entity ? entity : `<mark>${word}</mark>`));
  }

  /* ---------- streaming and errors ---------- */

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
    if (status === 429) return "Too many requests. Please wait a moment and try again.";
    return `Request failed (HTTP ${status}).`;
  }

  /* ---------- uploads, history, labels ---------- */

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

  const SUGGESTIONS = ["Summarize this document", "What are the key points?", "List the main topics covered"];

  function plural(count, word) {
    return `${count} ${word}${count === 1 ? "" : "s"}`;
  }

  function formatBytes(n) {
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(n < 10240 ? 1 : 0)} KB`;
    return `${(n / 1024 / 1024).toFixed(1)} MB`;
  }

  /** "0.4 s", "12.3 s", "1 min 5 s" */
  function formatDuration(ms) {
    if (!(ms >= 0)) return "";
    if (ms < 50) return "<0.1 s";
    const tenths = Math.round(ms / 100);
    if (tenths < 600) return `${(tenths / 10).toFixed(1)} s`;
    const total = Math.round(ms / 1000);
    return `${Math.floor(total / 60)} min ${total % 60} s`;
  }

  /** Footer under a finished answer: how long it took and what it used. */
  function answerSummary(meta, totalMs, firstWordMs) {
    const parts = [formatDuration(totalMs)];
    if (firstWordMs != null && firstWordMs < totalMs - 400) parts.push(`first word ${formatDuration(firstWordMs)}`);
    if (meta && meta.mode === "documents" && meta.sources) parts.push(plural(meta.sources, "passage"));
    return parts.filter(Boolean).join(" · ");
  }

  /** "just now", "5 min ago", "3 h ago", "2 days ago", otherwise the date. `now` is injectable for tests. */
  function timeAgo(iso, now = Date.now()) {
    const then = Date.parse(iso);
    if (Number.isNaN(then)) return "";
    const seconds = Math.max(0, Math.round((now - then) / 1000));
    if (seconds < 60) return "just now";
    if (seconds < 3600) return `${Math.floor(seconds / 60)} min ago`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)} h ago`;
    if (seconds < 86400 * 7) return `${Math.floor(seconds / 86400)} days ago`.replace("1 days", "1 day");
    return new Date(then).toISOString().slice(0, 10);
  }

  return {
    FALLBACK_PREFIX, MAX_UPLOAD_BYTES, HISTORY_MAX_MESSAGES, MODE_HELP, SUGGESTIONS,
    escapeHtml, stripFallback, renderAnswer, queryTerms, highlight, parseSSE, apiErrorMessage,
    validateUpload, formatBytes, plural, buildHistory, modeBadge, formatDuration, answerSummary, timeAgo,
  };
});
