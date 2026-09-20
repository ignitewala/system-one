"""
03_tool-gating.py — Jev in the agent loop: route the model, gate the tool.
"""
from dotenv import load_dotenv
load_dotenv()

from langchain_typesafe import TypeSafeClassifier, Choice, Score, Noul

FAST, POWERFUL = "gpt-4o-mini", "gpt-4o"

REQUESTS = [
    "What is the capital of France?",
    "Rename variable x to row_count in utils.py.",
    "Redesign payment reconciliation to survive a two-region outage.",
]

# --- 1. Which model should handle this request? ---
router = TypeSafeClassifier(
    questions={
        "model": Choice(
            instructions="Choose the least costly model that can do this task.",
            criteria={
                "fast": "Lookups, extraction, small localized edits.",
                "powerful": "Architecture, multi-step reasoning, high stakes.",
            },
        )
    },
)

for req in REQUESTS:
    c = router.invoke(req).choices["model"]
    picked = FAST if c.choice == "fast" else POWERFUL
    print(f"{req[:45]:47} -> {picked:14} {c.probabilities}")

# --- 2. Is this command safe to run? ---
gate = TypeSafeClassifier(
    questions={
        "risk": Score(
            instructions="How risky is this shell command?",
            criteria=["Reversible", "Needs care", "Destructive"],
        ),
        "approve": Noul(
            instructions="A human should approve this before it runs."
        ),
    },
)

print()
for cmd in ["ls -la", "rm -rf ./build", "git push --force origin main", "DROP TABLE claims;"]:
    r = gate.invoke(cmd)
    risk = r.scores["risk"].score
    approve = r.nouls["approve"].noul
    print(f"{cmd:32} risk={risk:.2f} approve={approve:.2f} "
          f"{'BLOCK' if approve > 0.7 else 'allow'}")

# --- 3. Same routing, as agent middleware ---
from langchain.agents import create_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_typesafe.experimental.middleware import (
    ModelRouterMiddleware, ModelChoice,
)

class Spy(BaseCallbackHandler):
    """Reports which model actually served the call."""
    def on_chat_model_start(self, serialized, messages, **kwargs):
        meta = kwargs.get("metadata") or {}
        print(f"    served by: {meta.get('ls_model_name', '?')}")

agent = create_agent(
    f"openai:{FAST}",
    middleware=[ModelRouterMiddleware(
        choices={
            "fast": ModelChoice(model=f"openai:{FAST}",
                                criteria="Lookups, extraction, small edits."),
            "powerful": ModelChoice(model=f"openai:{POWERFUL}",
                                    criteria="Architecture and high stakes."),
        },
        instructions="Choose the least costly model that can do this task.",
    )],
)

print()
for req in REQUESTS[::2]:          # one easy, one hard
    print(f"{req[:60]}")
    out = agent.invoke({"messages": [{"role": "user", "content": req}]},
                       config={"callbacks": [Spy()]})
    print(f"    {out['messages'][-1].content[:90]}")
