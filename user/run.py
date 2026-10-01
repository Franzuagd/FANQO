"""Minimal invariant-construction experiment."""

import fanqo as fq


def main():
    fq.load("general_config.py", force=True)

    print(fq.status())

    # Build any constructor directly.
    for method in fq.available_methods():
        result = fq.construct(method)
        print(method, result["details"])

    # name1 is red when better; name2 is blue when better.
    # Full 5-D comparisons:
    fq.compare("a_box", "eigen")
    fq.compare("graded_ls", "eigen")
    fq.compare("a_box", "graded_ls")
    fq.compare("hybrid", "graded_ls")

    # If either method is a_box_y0, compare() automatically tracks y0=0 only.
    fq.compare("a_box_y0", "eigen")
    fq.compare("a_box_y0", "hybrid")
    fq.compare("a_box_y0", "graded_ls")


if __name__ == "__main__":
    main()
