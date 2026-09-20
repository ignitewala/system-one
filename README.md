# system-one
System-one model evaluation and example

Runnable examples for [Jev](https://typesafe.ai), the first System One model from TypeSafe AI.

A System One model does not generate text. You give it a **state** (a string or a JSON object) and a set of **typed questions**. It evaluates every question in parallel in a single pass and returns typed answers with probabilities. There is no string to parse and no JSON to validate.

Three question primitives:

| Primitive | Ask it | You get back |
|---|---|---|
| `Choice` | Pick one of my options | the chosen option + a probability for each |
| `Score` | Place this on my ordered scale | a continuous value across your levels |
| `Noul` | How true is this statement | a float from 0 to 1 |

What it cannot do: chat, generate text, write code, or run an agent loop. It decides, scores and routes. That is the whole menu.

---

## Install

Requires Python 3.10+.

```bash
git clone https://github.com/ignitewala/system-one.git
cd system-one

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt`:

```
langchain
langchain-typesafe
langchain-openai
openai
python-dotenv
scikit-learn
```

### Keys

Copy the template and fill in your keys:

```bash
cp .env.example .env
```

```
TYPESAFE_API_KEY=your_typesafe_key
OPENAI_API_KEY=your_openai_key
```

Get a TypeSafe key at [typesafe.ai](https://typesafe.ai). The OpenAI key is only needed for the latency comparison in example 1 and the middleware section of example 3.



---

## The API shape, in one line

Questions are the **schema** and are bound at construction. State is the **input** and is passed at invoke.

```python
clf = TypeSafeClassifier(questions={...})   # build once
resp = clf.invoke(state)                    # call many times
```

This is closer to sklearn's `fit` / `predict` split than to an LLM call. A Jev classifier is a declared, reusable component inside ordinary software, not a conversation.

---

## Example 1 — `01_iris_classifier.py`

**Classic classifier vs System One vs frontier LLM, on the same problem.**

```bash
python 01_iris_classifier.py
```

Three ways to answer "which iris species is this?":

- **sklearn LogisticRegression** — needs four clean floats. Sub-millisecond, free, and completely unable to read a sentence.
- **Jev** — takes the sentence a botanist actually wrote, returns a typed species plus a probability for each option.
- **gpt-4o-mini** — same description, same options, prompted for JSON. The answer arrives as a string you must strip, parse and validate.

The script warms up each endpoint (so TLS and DNS are not charged to the model), runs five timed calls, and reports the median. It prints a comparison table of answer, latency, input tokens, output tokens and cost per million calls.

Two things to watch:

The JSON parse block around the OpenAI response has no counterpart on the Jev side. A `Choice` is structurally always one of your three options.

Output tokens are where the cost separates. Jev charges nothing for them.

Wall-clock timings include network round-trip to whichever region each provider terminates in, so some of the gap is geography rather than architecture. Run it a few times across the day before quoting a number.

---

## Example 2 — `02_parallel_questions.py`

**One state, three different question types, one request.**

```bash
python 02_parallel_questions.py
```

A support ticket goes in as a JSON object. Three questions come back at once:

- `team` — a **Choice** between billing, technical and other
- `severity` — a **Score** across cosmetic, workaround exists, and blocking
- `wants_refund` — a **Noul**, the probability the customer is asking for money back

Every question in a request is evaluated in parallel. Adding a question barely moves latency and costs only the tokens for that question's text.

The important constraint: **the questions do not see each other.** `severity` has no idea what `team` answered. That independence is the precondition for packing them into one call. If question B genuinely depends on the answer to A, that is a second call with a new state, not a conversation.

Note also that `Noul` instructions are written as a **statement**, not a question — "The customer is asking for money back." rather than "Is the customer asking for money back?" The model is scoring how true an assertion is.

Example -
![Uploading image.png…]()

---

## Example 3 — `03_tool-gating.py`

**Jev inside the agent loop: route the model, gate the tool.**

```bash
python 03_tool-gating.py
```

An agent runs the same cycle repeatedly — the LLM decides, a tool executes, something evaluates the result, repeat. Every one of those decisions is normally another full model call. This is where a cheap, fast, typed classifier earns its place.

Three parts:

**1. Model routing.** Three requests of very different difficulty are put to a `Choice` between a fast model and a powerful one. A capital-city lookup and an architecture redesign should not cost the same. The probability distribution is printed alongside the pick, so you can see how confident the router actually was.

**2. Tool risk gating.** Four shell commands are scored for risk and for whether a human should approve them. Claude Code, Codex and Cursor all ship a classifier like this to catch dangerous actions before they execute — until now it lived inside the closed part of the harness.

Worth checking in the output: whether `DROP TABLE claims;` scores meaningfully above `rm -rf ./build`. One destroys a build artifact, the other destroys production data, and a useful risk model should separate them.

**3. The same routing as middleware.** `ModelRouterMiddleware` wired into a real agent, with a small callback handler reporting which model actually served each call. Part 1 predicts, part 3 confirms. If they ever disagree, that disagreement is a real finding.

Parts 1 and 2 need only a TypeSafe key. Part 3 also calls OpenAI.

Example -
<img width="953" height="310" alt="image" src="https://github.com/user-attachments/assets/e05ff88c-586d-4501-a407-28da949967de" />

