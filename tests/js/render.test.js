const test = require("node:test");
const assert = require("node:assert/strict");
const L = require("../../docqa/static/docqa/lib.js");

/** Small seeded PRNG so "random" tests are repeatable. */
function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const pick = (r, list) => list[Math.floor(r() * list.length)];

/* ---------- markdown-ish rendering ---------- */

test("bullet lists, with * or - or a bullet character, become one <ul>", () => {
  const html = L.renderAnswer("Intro:\n* one\n* **two** [1]\n- three\n\nAfter", 2);
  assert.equal(
    html,
    '<p>Intro:</p><ul><li>one</li><li><strong>two</strong> <button type="button" class="cite" data-n="1" title="Show source 1">1</button></li><li>three</li></ul><p>After</p>'
  );
});

test("numbered lists accept 1. and 1) and stay separate from bullet lists", () => {
  assert.equal(L.renderAnswer("1. a\n2) b", 0), "<ol><li>a</li><li>b</li></ol>");
  assert.equal(L.renderAnswer("- a\n1. b", 0), "<ul><li>a</li></ul><ol><li>b</li></ol>");
});

test("numbers that are not list markers stay in paragraphs", () => {
  for (const text of ["3.5 million users", "2024. Revenue grew", "1.5x faster", "**Bold:** starts the line", "a - b - c"]) {
    assert.ok(L.renderAnswer(text, 0).startsWith("<p>"), text);
  }
});

test("fenced code blocks are left exactly as written: no bold, no citations, escaped", () => {
  const html = L.renderAnswer("Use:\n```py\nx = items[1]  # **not bold** <b>\n```\ndone [1]", 3);
  assert.ok(html.includes("<pre><code>x = items[1]  # **not bold** &lt;b&gt;</code></pre>"));
  assert.ok(html.includes('data-n="1"') && html.endsWith("</p>"));
  assert.ok(!html.slice(html.indexOf("<pre>"), html.indexOf("</pre>")).includes("cite"));
});

test("an unclosed code fence (still streaming) runs to the end instead of breaking the page", () => {
  const html = L.renderAnswer("Here:\n```\nline one\nline two", 0);
  assert.equal(html, "<p>Here:</p><pre><code>line one\nline two</code></pre>");
});

test("headings become small headings; inline code keeps citations literal", () => {
  assert.equal(L.renderAnswer("## Summary", 0), "<h4>Summary</h4>");
  assert.equal(L.renderAnswer("see `a[1]` and [1]", 1), 'see <code>a[1]</code> and <button type="button" class="cite" data-n="1" title="Show source 1">1</button>'.replace(/^/, "<p>").concat("</p>"));
});

test("CRLF line endings and empty input are handled", () => {
  assert.equal(L.renderAnswer("a\r\n\r\nb", 0), "<p>a</p><p>b</p>");
  assert.equal(L.renderAnswer("", 0), "");
  assert.equal(L.renderAnswer("   \n\n  ", 0), "");
});

test("renderAnswer output is always well formed and never contains an unescaped tag (random input)", () => {
  const r = rng(2026);
  const pieces = ["<", ">", "&", '"', "'", "`", "```", "``", "**", "*", "* ", "- ", "1. ", "2) ", "# ", "## ", "[1]", "[2, 3]", "[", "]", "\n", "\n\n",
    "<script>alert(1)</script>", "<img src=x onerror=alert(1)>", "</p>", "<p>", "javascript:alert(1)", " ", "  ", "a", "word", "é", "\u{1F600}", " ", "&lt;", "&amp;", "\\"];
  const allowed = /<\/?(?:p|br|code|pre|strong|ul|ol|li|h4)>|<button type="button" class="cite" data-n="\d+" title="Show source \d+">\d+<\/button>/g;
  for (let n = 0; n < 4000; n++) {
    const text = Array.from({ length: 1 + Math.floor(r() * 40) }, () => pick(r, pieces)).join("");
    const html = L.renderAnswer(text, Math.floor(r() * 4));
    const leftover = html.replace(allowed, "");
    assert.ok(!/[<>]/.test(leftover), `unexpected markup for ${JSON.stringify(text)} -> ${html}`);
    for (const tag of ["p", "pre", "code", "strong", "ul", "ol", "li", "h4"]) {
      const opens = (html.match(new RegExp(`<${tag}>`, "g")) || []).length;
      const closes = (html.match(new RegExp(`</${tag}>`, "g")) || []).length;
      assert.equal(opens, closes, `unbalanced <${tag}> for ${JSON.stringify(text)} -> ${html}`);
    }
  }
});

/* ---------- highlighting ---------- */

test("highlight marks whole words case-insensitively, allows a plural, and escapes everything else", () => {
  assert.equal(L.highlight("Refunds are accepted. The refund window is 30 days.", "refund window"),
    "<mark>Refunds</mark> are accepted. The <mark>refund</mark> <mark>window</mark> is 30 days.");
  assert.equal(L.highlight("an article about art", "art"), "an article about <mark>art</mark>");
  assert.equal(L.highlight("<b>refund</b> & more", "refund"), "&lt;b&gt;<mark>refund</mark>&lt;/b&gt; &amp; more");
});

test("highlight never breaks HTML entities and ignores stop words and short words", () => {
  assert.equal(L.highlight("fish & chips", "amp fish"), "<mark>fish</mark> &amp; chips");
  assert.equal(L.highlight("what is the refund", "what is the"), "what is the refund");
  assert.equal(L.highlight("go to it", "go to it"), "go to it");
  assert.equal(L.highlight("text", ""), "text");
  assert.equal(L.highlight("text", null), "text");
});

test("queryTerms keeps at most eight distinct meaningful words", () => {
  const terms = L.queryTerms("What is the refund policy? Refund, refunds, shipping warranty returns exchange store credit gift cards online");
  assert.ok(terms.length <= 8 && terms.includes("refund") && !terms.includes("what") && new Set(terms).size === terms.length);
});

test("highlight is safe for any text and query, including regex characters", () => {
  const r = rng(99);
  const bits = ["a", "bc", "refund", "c++", "(x|y)", "[z]", "\\", ".*", "$^", "<i>", "&amp;", "&", ";", "#39", " ", "\n", "é", "\u{1F600}", "s", "'"];
  for (let n = 0; n < 3000; n++) {
    const text = Array.from({ length: 1 + Math.floor(r() * 20) }, () => pick(r, bits)).join("");
    const query = Array.from({ length: Math.floor(r() * 8) }, () => pick(r, bits)).join(" ");
    const html = L.highlight(text, query);
    assert.ok(!/[<>]/.test(html.replace(/<\/?mark>/g, "")), `${JSON.stringify(text)} / ${JSON.stringify(query)} -> ${html}`);
  }
});

/* ---------- the stream parser ---------- */

test("parseSSE gives the same events however the stream is cut up (random events, random cuts)", () => {
  const r = rng(7);
  const payloads = [{ t: "plain" }, { t: 'quo"te' }, { t: "line\nbreak\n\ndouble" }, { t: "\\back" }, { t: "\u{1F600} é 中" }, { sources: [{ n: 1, text: "x\ny" }] }, {}];
  const names = ["sources", "token", "mode", "done", "error"];
  for (let round = 0; round < 300; round++) {
    const sent = Array.from({ length: 1 + Math.floor(r() * 6) }, () => ({ event: pick(r, names), data: pick(r, payloads) }));
    let wire = "";
    for (const { event, data } of sent) {
      if (r() < 0.2) wire += ": keep-alive\n\n";
      wire += `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
    }
    if (r() < 0.3) wire = wire.replace(/\n/g, "\r\n");
    let buffer = "";
    const got = [];
    for (let i = 0; i < wire.length; ) {
      const step = 1 + Math.floor(r() * 9);
      buffer += wire.slice(i, i + step);
      i += step;
      const parsed = L.parseSSE(buffer);
      buffer = parsed.rest;
      got.push(...parsed.events);
    }
    assert.deepEqual(got, sent, `round ${round}`);
  }
});

test("parseSSE never throws on garbage", () => {
  const r = rng(3);
  const bits = ["event:", "data:", "\n", "\n\n", "{", "}", '"', ":", " ", "\r", "x", "[", "null", "\u0000"];
  for (let n = 0; n < 2000; n++) {
    const text = Array.from({ length: Math.floor(r() * 30) }, () => pick(r, bits)).join("");
    const out = L.parseSSE(text);
    assert.ok(Array.isArray(out.events) && typeof out.rest === "string");
  }
});

/* ---------- history, durations, dates ---------- */

test("buildHistory survives junk and stays bounded", () => {
  const r = rng(5);
  const junk = [null, undefined, 5, "text", [], {}, { role: "user" }, { role: "user", content: 5 }, { role: "system", content: "x" },
    { role: "user", content: "hi" }, { role: "assistant", content: "yo" }, { role: "user", content: "x".repeat(4000) }, { content: "no role" }];
  for (let n = 0; n < 500; n++) {
    const chat = Array.from({ length: Math.floor(r() * 30) }, () => pick(r, junk));
    const history = L.buildHistory(chat);
    assert.ok(history.length <= L.HISTORY_MAX_MESSAGES);
    for (const m of history) {
      assert.ok(m.role === "user" || m.role === "assistant");
      assert.ok(typeof m.content === "string" && m.content.length > 0 && m.content.length <= 1500);
    }
  }
});

test("formatDuration", () => {
  assert.equal(L.formatDuration(0), "<0.1 s");
  assert.equal(L.formatDuration(49), "<0.1 s");
  assert.equal(L.formatDuration(400), "0.4 s");
  assert.equal(L.formatDuration(12345), "12.3 s");
  assert.equal(L.formatDuration(59949), "59.9 s");
  assert.equal(L.formatDuration(59999), "1 min 0 s");
  assert.equal(L.formatDuration(65000), "1 min 5 s");
  assert.equal(L.formatDuration(-5), "");
  assert.equal(L.formatDuration(NaN), "");
  assert.equal(L.formatDuration(undefined), "");
});

test("answerSummary shows time, first word only when it differs, and passages only for document answers", () => {
  assert.equal(L.answerSummary({ mode: "documents", sources: 4 }, 4100, 1600), "4.1 s · first word 1.6 s · 4 passages");
  assert.equal(L.answerSummary({ mode: "documents", sources: 1 }, 1200, 1100), "1.2 s · 1 passage");
  assert.equal(L.answerSummary({ mode: "general", sources: 0 }, 3000, 1000), "3.0 s · first word 1.0 s");
  assert.equal(L.answerSummary(null, 2000, null), "2.0 s");
});

test("timeAgo", () => {
  const now = Date.parse("2026-11-01T12:00:00Z");
  assert.equal(L.timeAgo("2026-11-01T11:59:40Z", now), "just now");
  assert.equal(L.timeAgo("2026-11-01T11:55:00Z", now), "5 min ago");
  assert.equal(L.timeAgo("2026-11-01T09:00:00Z", now), "3 h ago");
  assert.equal(L.timeAgo("2026-10-31T12:00:00Z", now), "1 day ago");
  assert.equal(L.timeAgo("2026-10-29T12:00:00Z", now), "3 days ago");
  assert.equal(L.timeAgo("2026-09-01T12:00:00Z", now), "2026-09-01");
  assert.equal(L.timeAgo("not a date", now), "");
  assert.equal(L.timeAgo("2026-11-01T12:30:00Z", now), "just now"); // a clock a little ahead is not "in the past"
});

test("rate-limit and unknown errors have readable messages", () => {
  assert.match(L.apiErrorMessage(429, null), /too many requests/i);
  assert.equal(L.apiErrorMessage(500, null), "Request failed (HTTP 500).");
});
