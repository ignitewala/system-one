from dotenv import load_dotenv
load_dotenv()

from langchain_typesafe import TypeSafeClassifier, Choice, Score, Noul

jev = TypeSafeClassifier(
    questions={
        "team": Choice(
            instructions="Which team should handle this?",
            criteria={
                "billing": "payment or charge problems",
                "technical": "bugs and outages",
                "other": "anything else",
            },
        ),
        "severity": Score(
            instructions="How severe is this?",
            criteria=["Cosmetic", "Workaround exists", "Blocking, no workaround"],
        ),
        "wants_refund": Noul(
            instructions="The customer is asking for money back."
        ),
    },
)

resp = jev.invoke(
    {
        "subject": "Charged twice for order #4417",
        "body": "My card was hit twice for one order. Six days, no reply.",
        "tier": "enterprise",
    }
)

team = resp.choices["team"]
severity = resp.scores["severity"]
refund = resp.nouls["wants_refund"]

print(f"\n\n Team:         {team.choice}")
print(f"  Probablities of choices:      {team.probabilities}")
print(f"  Confidence: {team.confidence}")
print(f"  Severity Score:     {severity.score:.2f}")
print(f"wants_refund: {refund.noul:.2f}")

print("\n\nraw:", resp)