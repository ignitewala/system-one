from dotenv import load_dotenv
load_dotenv()

import json
import time
import statistics

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from langchain_typesafe import TypeSafeClassifier, Choice
from openai import OpenAI

# ----------------------------------------------------------------------
# Shared context — identical text goes to both models
# ----------------------------------------------------------------------
STATE = (
    "Small flower, petals barely 1.4cm long and very narrow, "
    "sepals wide relative to length, grows in cold alpine soil."
)

CRITERIA = {
    "setosa": "very short narrow petals, wide sepals",
    "versicolor": "medium petals, intermediate everything",
    "virginica": "long broad petals, largest overall",
}

RUNS = 5  # median of N, not a single sample

# Published pricing, USD per 1M tokens. Update if these move.
PRICE = {
    "jev":          {"in": 0.042, "out": 0.0},
    "gpt-4o-mini":  {"in": 0.15,  "out": 0.60},
}


def timed(fn, runs=RUNS):
    """Warm up once (pays TLS/DNS), then time `runs` calls. Returns (median_ms, all_ms, last_result)."""
    fn()  # warm-up, discarded
    times, result = [], None
    for i in range(runs):
        t0 = time.perf_counter()
        result = fn()
        ms = (time.perf_counter() - t0) * 1000
        times.append(ms)
        print(f"    run {i+1}: {ms:7.1f} ms")
    return statistics.median(times), times, result


def cost_usd(model, tin, tout):
    p = PRICE[model]
    return (tin / 1e6) * p["in"] + (tout / 1e6) * p["out"]


# ======================================================================
# 1. Classic classifier — the baseline, for scale
# ======================================================================
print("\n=== 1. sklearn LogisticRegression ===")
X, y = load_iris(return_X_y=True)
clf = LogisticRegression(max_iter=200).fit(X, y)

t0 = time.perf_counter()
pred = clf.predict([[5.1, 3.5, 1.4, 0.2]])
sk_ms = (time.perf_counter() - t0) * 1000
print(f"  prediction: {pred}  ({['setosa','versicolor','virginica'][pred[0]]})")
print(f"  latency:    {sk_ms:.3f} ms   tokens: 0   cost: $0")
print("  note: needs 4 clean floats. Cannot read the sentence.")


# ======================================================================
# 2. Jev — System One
# ======================================================================
print("\n=== 2. Jev (System One) ===")
jev = TypeSafeClassifier(
    questions={
        "species": Choice(
            instructions="Which iris species is this most likely to be?",
            criteria=CRITERIA,
        )
    },
)

jev_ms, jev_all, jev_resp = timed(lambda: jev.invoke(STATE))

print("\n  raw response:")
print("  ", jev_resp)

# Defensive extraction — beta API, shape may shift
try:
    jev_answer = jev_resp.choices["species"].choice
    jev_probs = jev_resp.choices["species"].probabilities
except AttributeError:
    jev_answer, jev_probs = "<check raw above>", None

# Usage — attribute name may differ; inspect raw if this misses
jev_in = jev_out = 0
for attr in ("usage", "response_metadata", "token_usage"):
    u = getattr(jev_resp, attr, None)
    if u:
        print(f"  usage ({attr}):", u)
        if isinstance(u, dict):
            jev_in = u.get("input_tokens") or u.get("prompt_tokens") or 0
            jev_out = u.get("output_tokens") or u.get("completion_tokens") or 0
        break


# ======================================================================
# 3. OpenAI — same question, forced to answer in a parseable form
# ======================================================================
print("\n=== 3. OpenAI gpt-4o-mini ===")
client = OpenAI()

PROMPT = (
    f"Classify the following description into exactly one iris species.\n\n"
    f"Description: {STATE}\n\n"
    f"Options:\n"
    + "\n".join(f"- {k}: {v}" for k, v in CRITERIA.items())
    + "\n\nRespond with ONLY valid JSON: "
      '{"species": "<one of setosa|versicolor|virginica>", "confidence": <0.0-1.0>}'
      "\nNo markdown fences, no explanation."
)

def call_openai():
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": PROMPT}],
        temperature=0,
    )

oai_ms, oai_all, oai_resp = timed(call_openai)

raw_text = oai_resp.choices[0].message.content
print(f"\n  raw text returned: {raw_text!r}")

# This block is the point of the whole exercise — it does not exist for Jev
try:
    parsed = json.loads(raw_text.replace("```json", "").replace("```", "").strip())
    oai_answer = parsed["species"]
    if oai_answer not in CRITERIA:
        oai_answer = f"INVALID CATEGORY: {oai_answer}"
except (json.JSONDecodeError, KeyError) as e:
    oai_answer = f"PARSE FAILED: {e}"

oai_in = oai_resp.usage.prompt_tokens
oai_out = oai_resp.usage.completion_tokens


# ======================================================================
# 4. Comparison
# ======================================================================
print("\n" + "=" * 68)
print(f"{'':16}{'answer':>14}{'median ms':>12}{'tok in':>9}{'tok out':>9}{'$/1M calls':>12}")
print("-" * 68)
print(f"{'sklearn':16}{'setosa':>14}{sk_ms:>12.3f}{0:>9}{0:>9}{0:>12.2f}")
print(f"{'jev':16}{str(jev_answer):>14}{jev_ms:>12.1f}{jev_in:>9}{jev_out:>9}"
      f"{cost_usd('jev', jev_in, jev_out) * 1e6:>12.2f}")
print(f"{'gpt-4o-mini':16}{str(oai_answer):>14}{oai_ms:>12.1f}{oai_in:>9}{oai_out:>9}"
      f"{cost_usd('gpt-4o-mini', oai_in, oai_out) * 1e6:>12.2f}")
print("=" * 68)

if jev_ms and oai_ms:
    print(f"\n  Jev is {oai_ms / jev_ms:.1f}x faster (median wall-clock)")
if jev_in:
    c_j = cost_usd('jev', jev_in, jev_out)
    c_o = cost_usd('gpt-4o-mini', oai_in, oai_out)
    if c_j:
        print(f"  Jev is {c_o / c_j:.1f}x cheaper per call")

print(f"\n  spread — jev: {min(jev_all):.0f}-{max(jev_all):.0f} ms, "
      f"openai: {min(oai_all):.0f}-{max(oai_all):.0f} ms")
print(f"  jev probabilities: {jev_probs}")
