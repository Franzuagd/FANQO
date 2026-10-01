Ixcononly local workflow

1. Edit lattice_config.py only when the machine changes.
2. Edit general_config.py only for polynomial, a_box, tracking, or output settings.
3. Run:

   python run.py

Public invariant methods:
    a_box
    hybrid
    eigen
    graded_ls
    cesaro
    abel
    a_box_y0

a_box, hybrid, eigen, graded_ls, cesaro, and abel are full 5-D constructors.
graded_ls, cesaro, and abel are A_BOX-independent and keep the Sx interpretation;
cesaro and abel obtain the nonlinear correction by map averaging.

Main comparison:
    fq.compare("name1", "name2")

Red means name1 has smaller tracked Ix drift.
Blue means name2 has smaller tracked Ix drift.

There is no magnet optimization and no optimize_a_box() on this branch.
