from pathlib import Path
from .decoder import constrained_decode, unconstrained_decode
from .evaluation import evaluate, save_json
from .meter import Meter, prastara


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    target = Meter("LGGL")
    unconstrained = unconstrained_decode(lines=8)
    constrained = constrained_decode(target, lines=8)
    result = {
        "experiment": "pingala_constraint_v1",
        "prastara_length_4": prastara(4),
        "target_meter": target.pattern,
        "unconstrained": evaluate(unconstrained, target),
        "constrained": evaluate(constrained, target),
        "poems": {
            "unconstrained": [x.text for x in unconstrained],
            "constrained": [x.text for x in constrained],
        },
    }
    save_json(root / "results" / "experiment.json", result)
    print(result)


if __name__ == "__main__":
    main()
