from .decoder import constrained_decode, unconstrained_decode
from .meter import Meter


def show(name, poem):
    print(f"\n{name}")
    for line in poem:
        print(f"{line.pattern:>6}  {line.text}")


if __name__ == "__main__":
    target = Meter("LGGL")
    show("Unconstrained", unconstrained_decode(lines=4))
    show("Constrained to LGGL", constrained_decode(target, lines=4))
