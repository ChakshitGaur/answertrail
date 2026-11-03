import os
import statistics
import time

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from docqa.services import llm

QUESTION = "What is the refund window?"
PASSAGES = [
    "Refunds are accepted within 30 days of purchase, provided the item is unused and in its original packaging.",
    "Shipping takes 5 business days within the country and up to 14 days internationally.",
    "Gift cards cannot be refunded or exchanged for cash.",
    "To start a return, contact support with your order number. A prepaid label is emailed within one business day.",
]
_STOP_AFTER_FAILURES = 2  # a model that keeps failing (quota, bad name) is not asked again, so no quota is burned


class Command(BaseCommand):
    help = (
        "Measure how long each model takes to produce its first word on a typical question, so you can order "
        "LLM_MODEL / LLM_FALLBACK_MODELS (or compare providers) with numbers. Makes real model calls: a few per model."
    )

    def add_arguments(self, parser):
        parser.add_argument("--runs", type=int, default=3, help="calls per model (default 3)")
        parser.add_argument("--models", help="comma separated; default: LLM_MODEL then LLM_FALLBACK_MODELS")
        parser.add_argument("--base-url", help="measure another OpenAI-compatible server instead of LLM_BASE_URL")
        parser.add_argument(
            "--key-env", help="name of the environment variable holding that server's key (the key is never typed here)"
        )
        parser.add_argument("--pause", type=float, default=1.0, help="seconds between calls (default 1)")

    def handle(self, *args, **opts):
        if opts["base_url"]:
            settings.LLM_BASE_URL = opts["base_url"]
            key = os.environ.get(opts["key_env"], "") if opts["key_env"] else ""
            if opts["key_env"] and not key:
                raise CommandError(f"the environment variable {opts['key_env']} is empty or not set")
            settings.LLM_API_KEY = key
        if not settings.LLM_BASE_URL:
            raise CommandError("this measures OpenAI-compatible servers: set LLM_BASE_URL (or pass --base-url)")
        names = [n.strip() for n in opts["models"].split(",") if n.strip()] if opts["models"] else llm._models()
        if not names:
            raise CommandError("no model to measure: set LLM_MODEL or pass --models")
        if opts["runs"] < 1:
            raise CommandError("--runs must be at least 1")

        effort = llm._effort_for(True)
        messages = llm._as_messages(llm.build_messages(QUESTION, PASSAGES, None))
        self.stdout.write(
            f"{settings.LLM_BASE_URL}  reasoning effort: {effort or 'provider default'}  "
            f"{opts['runs']} call(s) per model; the first call of each includes opening the connection\n"
        )
        rows = [self._measure(name, effort, messages, opts["runs"], opts["pause"]) for name in names]
        self._report(rows)

    def _measure(self, name, effort, messages, runs, pause):
        firsts, totals, errors = [], [], []
        failures = 0
        self.stdout.write(f"{name} ", ending="")
        for i in range(runs):
            if i:
                time.sleep(pause)
            started = time.perf_counter()
            first = None
            try:
                for chunk in llm._chat_model(name, effort).stream(messages):
                    if first is None and str(chunk.text):
                        first = time.perf_counter() - started
                if first is None:
                    raise ValueError("no text in the reply")
            except llm._provider_errors() as exc:
                errors.append(f"{exc.__class__.__name__}: {' '.join(str(exc).split())[:80]}")
                failures += 1
                self.stdout.write("x", ending="")
                if failures >= _STOP_AFTER_FAILURES:
                    break
                continue
            failures = 0
            firsts.append(first)
            totals.append(time.perf_counter() - started)
            self.stdout.write(".", ending="")
        self.stdout.write("")
        return {"name": name, "firsts": firsts, "totals": totals, "errors": errors}

    def _report(self, rows):
        self.stdout.write(f"\n{'model':32} {'ok':>4}  {'first word: median':>18} {'min':>6} {'max':>6}  {'total: median':>13}")
        for r in rows:
            f = r["firsts"]
            if f:
                self.stdout.write(
                    f"{r['name']:32} {len(f):>4}  {statistics.median(f):>17.2f}s {min(f):>5.2f}s {max(f):>5.2f}s "
                    f" {statistics.median(r['totals']):>12.2f}s"
                )
            else:
                self.stdout.write(f"{r['name']:32} {0:>4}  {'no successful call':>18}")
            if r["errors"]:
                self.stdout.write(f"{'':32}       {len(r['errors'])} failed, first: {r['errors'][0]}")
        if any("400" in e for r in rows for e in r["errors"]):
            self.stdout.write(
                "\nHTTP 400 usually means the model name is wrong or the model rejects the reasoning effort setting "
                "(the app retries such a model without it; here try LLM_REASONING_EFFORT= empty)."
            )
        ranked = sorted((r for r in rows if r["firsts"]), key=lambda r: statistics.median(r["firsts"]))
        if ranked:
            self.stdout.write(
                "\nfastest first word first: " + ", ".join(r["name"] for r in ranked)
                + "\nput the one you want as LLM_MODEL and the others in LLM_FALLBACK_MODELS. Free tiers vary by the "
                "hour, so repeat the measurement before relying on one run."
            )
        else:
            self.stdout.write("\nno model answered: check the key, the base URL and the model names above.")
