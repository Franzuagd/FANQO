"""General linear-optics tools.

Magnet representation used internally:
    [name, type, length, angle, K, S, O, M, M5]

Linear-data representation returned by linear_optics():
    [cs0, disp0, M4, M5, tune_x, tune_y, chrom_x, chrom_y,
     emittance, circumference, s_values, cs_values, disp_values]
"""
import math
import numpy as np

# NAME CALLINGS
def _magnet_field_index(field):  #field name to integer
    names = ("NAME", "TYPE", "LENGTH", "ANGLE", "K", "S", "O", "M", "M5")
    try:
        return names.index(str(field).upper())
    except ValueError as exc:
        raise KeyError(f"Unknown magnet field: {field}") from exc
def _linear_data_index(field): #linear data to integer
    names = (
        "CS0", "DISP0", "LATTICE_M4", "LATTICE_M5", "TUNE_X", "TUNE_Y",
        "CHROM_X", "CHROM_Y", "EMITTANCE", "CIRCUMFERENCE", "S_VALUES",
        "CS_VALUES", "DISP_VALUES")
    aliases = {"CELL_M4": "LATTICE_M4", "CELL_M5": "LATTICE_M5"}
    key = aliases.get(str(field).upper(), str(field).upper())
    try:
        return names.index(key)
    except ValueError as exc:
        raise KeyError(f"Unknown linear-data field: {field}") from exc
def magnet_field(elem, field):  #return the value of that field in elem
    return elem[_magnet_field_index(field)]
def set_magnet_field(elem, field, value): #Allows you to edit one of the magnects fields
    elem[_magnet_field_index(field)] = value
def linear_data(data, field):  #recover info from data by using names
    return data[_linear_data_index(field)] 
# LINEAR TRANSFER MATRICES

def plane_matrix(L, k):
    """Generic 2x2 uncoupled linear map for constant focusing k."""
    if abs(k) < 1e-15:
        return np.array([[1.0, L], [0.0, 1.0]])

    if k > 0.0:
        root = math.sqrt(k)
        C = math.cos(root * L)
        Sine = math.sin(root * L) / root
    else:
        root = math.sqrt(-k)
        C = math.cosh(root * L)
        Sine = math.sinh(root * L) / root

    return np.array([[C, Sine], [-k * Sine, C]])

def drift4(L):
    matrix = np.eye(4)
    matrix[0:2, 0:2] = plane_matrix(L, 0.0)
    matrix[2:4, 2:4] = plane_matrix(L, 0.0)
    return matrix

def drift5(L):
    matrix = np.eye(5)
    matrix[:4, :4] = drift4(L)
    return matrix

def quadrupole4(L, k):
    matrix = np.zeros((4, 4))
    matrix[0:2, 0:2] = plane_matrix(L, k)
    matrix[2:4, 2:4] = plane_matrix(L, -k)
    return matrix

def quadrupole5(L, k):
    matrix = np.eye(5)
    matrix[:4, :4] = quadrupole4(L, k)
    return matrix

def bending4(L, k, h):
    matrix = np.zeros((4, 4))
    matrix[0:2, 0:2] = plane_matrix(L, k + h**2)
    matrix[2:4, 2:4] = plane_matrix(L, -k)
    return matrix

def bending5(L, k, h):
    matrix = np.eye(5)
    matrix[:4, :4] = bending4(L, k, h)

    kx = k + h**2
    Cx = matrix[0, 0]
    Sx = matrix[0, 1]

    if abs(kx) < 1e-15:
        matrix[0, 4] = 0.5 * h * L**2
    else:
        matrix[0, 4] = h * (1.0 - Cx) / kx

    matrix[1, 4] = h * Sx
    return matrix

def transfer4(elem, L=None): #4x4 map through a complete magnet or through a piece of it.
    ds = elem[2] if L is None else float(L)

    if elem[1] in ("drift", "sextupole"):
        return drift4(ds)

    if elem[1] == "quadrupole":
        return quadrupole4(ds, elem[4])

    if elem[1] == "bending":
        if elem[2] == 0.0:
            return np.eye(4)
        h = math.radians(elem[3]) / elem[2]
        return bending4(ds, elem[4], h)

    if elem[1] == "multipole":
        return np.eye(4)

    raise ValueError("Unknown magnet type: " + str(elem[1]))

def transfer5(elem, L=None): #5x5 map used to propagate horizontal dispersion.
    ds = elem[2] if L is None else float(L)

    if elem[1] in ("drift", "sextupole"):
        return drift5(ds)

    if elem[1] == "quadrupole":
        return quadrupole5(ds, elem[4])

    if elem[1] == "bending":
        if elem[2] == 0.0:
            return np.eye(5)
        h = math.radians(elem[3]) / elem[2]
        return bending5(ds, elem[4], h)

    if elem[1] == "multipole":
        return np.eye(5)

    raise ValueError("Unknown magnet type: " + str(elem[1]))

def compute_matrices(elem): #Compute and store the full 4x4 and 5x5 linear maps of one magnet.
    elem[7] = transfer4(elem)
    elem[8] = transfer5(elem)


def unique_magnets(lattice): #Return each magnet object in a lattice only once, preserving order.
    result = []
    seen = set()
    for elem in lattice:
        marker = id(elem)
        if marker not in seen:
            seen.add(marker)
            result.append(elem)
    return result

def refresh_matrices(lattice): #Recompute M and M5 for all unique magnets in a lattice or magnet list.
    for elem in unique_magnets(lattice):
        compute_matrices(elem)

# LATTICE CONSTRUCTION AND EDITING
def build_lattice(names, magnets): #Replace every lattice name by its magnet list.
    magnet_by_name = {elem[0]: elem for elem in magnets}

    missing = [name for name in names if name not in magnet_by_name]
    if missing:
        raise KeyError("Undefined magnets: " + str(sorted(set(missing))))

    return [magnet_by_name[name] for name in names]

def get_magnet(lattice, magnet_name): #Return the first magnet with the requested name.
    for elem in lattice:
        if elem[0] == magnet_name:
            return elem
    raise KeyError(f"Magnet {magnet_name} was not found in the lattice.")


def set_quadrupole_strength(lattice, magnet_name, strength): #Update K and immediately refresh the corresponding linear matrices.
    elem = get_magnet(lattice, magnet_name)
    if elem[1] != "quadrupole":
        raise ValueError(f"{magnet_name} is not a quadrupole.")
    elem[4] = float(strength)
    compute_matrices(elem)


def set_sextupole_strength(lattice, family_name, strength): #Update S. No M/M5 refresh is needed because its linear map is a drift.
    elem = get_magnet(lattice, family_name)
    if elem[1] != "sextupole":
        raise ValueError(f"{family_name} is not a sextupole.")
    elem[5] = float(strength)


# LINEAR OPTICS
def courant_snyder_matrix(matrix):
    return np.array([
        [matrix[0, 0]**2, -2*matrix[0, 0]*matrix[0, 1], matrix[0, 1]**2, 0, 0, 0],
        [-matrix[0, 0]*matrix[1, 0], matrix[0, 0]*matrix[1, 1] + matrix[0, 1]*matrix[1, 0], -matrix[0, 1]*matrix[1, 1], 0, 0, 0],
        [matrix[1, 0]**2, -2*matrix[1, 0]*matrix[1, 1], matrix[1, 1]**2, 0, 0, 0],
        [0, 0, 0, matrix[2, 2]**2, -2*matrix[2, 2]*matrix[2, 3], matrix[2, 3]**2],
        [0, 0, 0, -matrix[2, 2]*matrix[3, 2], matrix[2, 2]*matrix[3, 3] + matrix[2, 3]*matrix[3, 2], -matrix[2, 3]*matrix[3, 3]],
        [0, 0, 0, matrix[3, 2]**2, -2*matrix[3, 2]*matrix[3, 3], matrix[3, 3]**2],
    ])

def propagate_linear_functions(lattice, cs0, disp0, step):
    s = 0.0
    mux = 0.0
    muy = 0.0
    chromx = 0.0
    chromy = 0.0
    radiation = np.zeros(6)

    cs = cs0.copy()
    disp = disp0.copy()

    s_values = [s]
    cs_values = [cs.copy()]
    disp_values = [disp.copy()]

    for elem in lattice:
        if elem[2] == 0.0:
            continue

        n_full = int(elem[2] // step)
        pieces = [step] * n_full
        remainder = elem[2] - n_full * step

        if remainder > 1e-14:
            pieces.append(remainder)

        for ds in pieces:
            beta_x_i = cs[0]
            beta_y_i = cs[3]

            k_eff = elem[4] - 2.0 * disp[0] * elem[5]
            if k_eff != 0.0:
                chromx += ds * beta_x_i * k_eff
                chromy += ds * beta_y_i * k_eff

            if elem[1] == "bending" and elem[3] != 0.0:
                h = math.radians(elem[3]) / elem[2]
                Hx = (
                    beta_x_i * disp[1]**2
                    + 2.0 * cs[1] * disp[0] * disp[1]
                    + cs[2] * disp[0]**2
                )

                radiation[0] += ds * disp[0] * h
                radiation[1] += ds * h**2
                radiation[2] += ds * abs(h)**3
                radiation[3] += ds * (1.0 + 2.0 * elem[4] / h**2) * disp[0] * h**3
                radiation[4] += ds * Hx * abs(h)**3
                radiation[5] += ds * (disp[0] * elem[4])**2

            matrix4 = transfer4(elem, ds)
            matrix5 = transfer5(elem, ds)

            cs = courant_snyder_matrix(matrix4) @ cs
            disp = matrix5 @ disp

            beta_x_f = cs[0]
            beta_y_f = cs[3]

            argx = matrix4[0, 1] / math.sqrt(beta_x_i * beta_x_f)
            argy = matrix4[2, 3] / math.sqrt(beta_y_i * beta_y_f)

            mux += math.asin(np.clip(argx, -1.0, 1.0))
            muy += math.asin(np.clip(argy, -1.0, 1.0))

            s += ds
            s_values.append(s)
            cs_values.append(cs.copy())
            disp_values.append(disp.copy())

    return [
        mux,
        muy,
        np.array(s_values),
        np.array(cs_values),
        np.array(disp_values),
        radiation,
        chromx,
        chromy,
    ]

def lattice_matrix(lattice):
    matrix4 = np.eye(4)
    matrix5 = np.eye(5)

    for elem in lattice:
        matrix4 = elem[7] @ matrix4
        matrix5 = elem[8] @ matrix5

    return matrix4, matrix5

def periodic_twiss_and_dispersion(lattice):
    matrix4, matrix5 = lattice_matrix(lattice)

    trace_x = matrix4[0, 0] + matrix4[1, 1]
    trace_y = matrix4[2, 2] + matrix4[3, 3]

    if abs(trace_x) >= 2.0 or abs(trace_y) >= 2.0:
        raise ValueError(
            f"Unstable lattice: horizontal trace={trace_x:.8f}, vertical trace={trace_y:.8f}"
        )

    sin_mux = np.sign(matrix4[0, 1]) * math.sqrt(
        -matrix4[0, 1] * matrix4[1, 0]
        - (matrix4[0, 0] - matrix4[1, 1])**2 / 4.0
    )

    sin_muy = np.sign(matrix4[2, 3]) * math.sqrt(
        -matrix4[2, 3] * matrix4[3, 2]
        - (matrix4[2, 2] - matrix4[3, 3])**2 / 4.0
    )

    ax = (matrix4[0, 0] - matrix4[1, 1]) / (2.0 * sin_mux)
    bx = matrix4[0, 1] / sin_mux
    gx = (1.0 + ax**2) / bx

    ay = (matrix4[2, 2] - matrix4[3, 3]) / (2.0 * sin_muy)
    by = matrix4[2, 3] / sin_muy
    gy = (1.0 + ay**2) / by

    denominator = 2.0 - matrix5[0, 0] - matrix5[1, 1]

    disp = (
        matrix5[0, 1] * matrix5[1, 4]
        + matrix5[0, 4] * (1.0 - matrix5[1, 1])
    ) / denominator

    dispd = (
        matrix5[1, 0] * matrix5[0, 4]
        + matrix5[1, 4] * (1.0 - matrix5[0, 0])
    ) / denominator

    cs0 = np.array([bx, ax, gx, by, ay, gy])
    disp0 = np.array([disp, dispd, 0.0, 0.0, 1.0])

    return cs0, disp0, matrix4, matrix5

def linear_optics(lattice, energy, repetitions, step):
    cs0, disp0, matrix4, matrix5 = periodic_twiss_and_dispersion(lattice)

    mux, muy, s_values, cs_values, disp_values, radiation, chromx_i, chromy_i = (
        propagate_linear_functions(lattice, cs0, disp0, step)
    )

    I2 = radiation[1]
    I4 = radiation[3]
    I5 = radiation[4]

    if I2 == 0.0 or I2 - I4 == 0.0:
        natural_emittance = float("nan")
    else:
        natural_emittance = (
            3.8319e-13
            * (1000.0 * energy / 0.5109989)**2
            * I5 / (I2 - I4)
        )

    tune_x = repetitions * mux / (2.0 * math.pi)
    tune_y = repetitions * muy / (2.0 * math.pi)
    chrom_x = -(repetitions / (4.0 * math.pi)) * chromx_i
    chrom_y = +(repetitions / (4.0 * math.pi)) * chromy_i
    circumference = repetitions * s_values[-1]

    return [
        cs0,
        disp0,
        matrix4,
        matrix5,
        tune_x,
        tune_y,
        chrom_x,
        chrom_y,
        natural_emittance,
        circumference,
        s_values,
        cs_values,
        disp_values,
    ]

# FIRST-ORDER CHROMATIC CORRECTION


def chromaticity_with_sextupoles(lattice, repetitions, step):
    """First-order chromaticities using K_eff = K - 2*Dx*S."""
    cs, disp, _, _ = periodic_twiss_and_dispersion(lattice)

    integral_x = 0.0
    integral_y = 0.0

    for elem in lattice:
        if elem[2] == 0.0:
            continue

        n_full = int(elem[2] // step)
        pieces = [step] * n_full
        remainder = elem[2] - n_full * step

        if remainder > 1e-14:
            pieces.append(remainder)

        for ds in pieces:
            k_eff = elem[4] - 2.0 * disp[0] * elem[5]

            integral_x += ds * cs[0] * k_eff
            integral_y += ds * cs[3] * k_eff

            cs = courant_snyder_matrix(transfer4(elem, ds)) @ cs
            disp = transfer5(elem, ds) @ disp

    chrom_x = -(repetitions / (4.0 * math.pi)) * integral_x
    chrom_y = +(repetitions / (4.0 * math.pi)) * integral_y

    return chrom_x, chrom_y


def correct_chromaticity(lattice, family1, family2, target_x, target_y, repetitions, step):
    """Solve the two-family first-order chromatic correction problem."""
    if family1 == family2:
        raise ValueError("family1 and family2 must be different sextupole families.")

    cs, disp, _, _ = periodic_twiss_and_dispersion(lattice)

    base_x = 0.0
    base_y = 0.0
    response_x1 = 0.0
    response_y1 = 0.0
    response_x2 = 0.0
    response_y2 = 0.0
    found1 = False
    found2 = False

    for elem in lattice:
        if elem[2] == 0.0:
            continue

        if elem[0] == family1:
            if elem[1] != "sextupole":
                raise ValueError(f"{family1} is not a sextupole.")
            found1 = True

        if elem[0] == family2:
            if elem[1] != "sextupole":
                raise ValueError(f"{family2} is not a sextupole.")
            found2 = True

        n_full = int(elem[2] // step)
        pieces = [step] * n_full
        remainder = elem[2] - n_full * step

        if remainder > 1e-14:
            pieces.append(remainder)

        for ds in pieces:
            beta_x = cs[0]
            beta_y = cs[3]
            Dx = disp[0]

            if elem[0] == family1:
                response_x1 += 2.0 * ds * beta_x * Dx
                response_y1 += 2.0 * ds * beta_y * Dx
                S_here = 0.0
            elif elem[0] == family2:
                response_x2 += 2.0 * ds * beta_x * Dx
                response_y2 += 2.0 * ds * beta_y * Dx
                S_here = 0.0
            else:
                S_here = elem[5]

            k_eff = elem[4] - 2.0 * Dx * S_here
            base_x += ds * beta_x * k_eff
            base_y += ds * beta_y * k_eff

            cs = courant_snyder_matrix(transfer4(elem, ds)) @ cs
            disp = transfer5(elem, ds) @ disp

    if not found1:
        raise KeyError(f"Sextupole family {family1} was not found in the lattice.")
    if not found2:
        raise KeyError(f"Sextupole family {family2} was not found in the lattice.")

    A = np.array([
        [response_x1, response_x2],
        [response_y1, response_y2],
    ])

    b = np.array([
        base_x + target_x * (4.0 * math.pi / repetitions),
        base_y - target_y * (4.0 * math.pi / repetitions),
    ])

    if abs(np.linalg.det(A)) < 1e-14:
        raise ValueError(
            "The two sextupole families do not provide independent chromatic correction."
        )

    S1, S2 = np.linalg.solve(A, b)
    set_sextupole_strength(lattice, family1, S1)
    set_sextupole_strength(lattice, family2, S2)

    corrected_x, corrected_y = chromaticity_with_sextupoles(
        lattice, repetitions=repetitions, step=step
    )

    return S1, S2, corrected_x, corrected_y


def correct_chromaticity_from_data( #Chromatic correction reusing optics already computed by linear_optics().
    lattice, data, family1, family2, target_x, target_y, repetitions, step
):
    if family1 == family2:
        raise ValueError("family1 and family2 must be different sextupole families.")

    cs_values = np.asarray(data[11], dtype=float)
    disp_values = np.asarray(data[12], dtype=float)

    base_x = base_y = 0.0
    response_x1 = response_y1 = 0.0
    response_x2 = response_y2 = 0.0
    found1 = found2 = False
    sample = 0

    for elem in lattice:
        if elem[2] == 0.0:
            continue

        if elem[0] == family1:
            if elem[1] != "sextupole":
                raise ValueError(f"{family1} is not a sextupole.")
            found1 = True
        if elem[0] == family2:
            if elem[1] != "sextupole":
                raise ValueError(f"{family2} is not a sextupole.")
            found2 = True

        n_full = int(elem[2] // step)
        pieces = [step] * n_full
        remainder = elem[2] - n_full * step
        if remainder > 1e-14:
            pieces.append(remainder)

        for ds in pieces:
            cs = cs_values[sample]
            disp = disp_values[sample]
            beta_x = cs[0]
            beta_y = cs[3]
            Dx = disp[0]

            if elem[0] == family1:
                response_x1 += 2.0 * ds * beta_x * Dx
                response_y1 += 2.0 * ds * beta_y * Dx
                S_here = 0.0
            elif elem[0] == family2:
                response_x2 += 2.0 * ds * beta_x * Dx
                response_y2 += 2.0 * ds * beta_y * Dx
                S_here = 0.0
            else:
                S_here = elem[5]

            k_eff = elem[4] - 2.0 * Dx * S_here
            base_x += ds * beta_x * k_eff
            base_y += ds * beta_y * k_eff
            sample += 1

    if sample != len(cs_values) - 1:
        raise RuntimeError(
            f"Linear sampling mismatch: used {sample} steps, "
            f"but data contains {len(cs_values) - 1}."
        )
    if not found1:
        raise KeyError(f"Sextupole family {family1} was not found in the lattice.")
    if not found2:
        raise KeyError(f"Sextupole family {family2} was not found in the lattice.")

    A = np.array([
        [response_x1, response_x2],
        [response_y1, response_y2],
    ])
    b = np.array([
        base_x + target_x * (4.0 * math.pi / repetitions),
        base_y - target_y * (4.0 * math.pi / repetitions),
    ])

    if abs(np.linalg.det(A)) < 1e-14:
        raise ValueError(
            "The two sextupole families do not provide independent chromatic correction."
        )

    S1, S2 = np.linalg.solve(A, b)
    set_sextupole_strength(lattice, family1, S1)
    set_sextupole_strength(lattice, family2, S2)

    corrected_integral_x = base_x - response_x1 * S1 - response_x2 * S2
    corrected_integral_y = base_y - response_y1 * S1 - response_y2 * S2
    corrected_x = -(repetitions / (4.0 * math.pi)) * corrected_integral_x
    corrected_y = +(repetitions / (4.0 * math.pi)) * corrected_integral_y

    return S1, S2, corrected_x, corrected_y


# PREPARATION / EFFICIENT UPDATE
def prepare_lattice(
    parameters,
    ring_names,
    magnet_builder,
    energy_parameter,
    correction_parameter_map,
    correct_chromatic,
    family1,
    family2,
    target_chrom_x,
    target_chrom_y,
    repetitions,
    step,
):
    """Build a configured ring and prepare its linear data."""
    p = parameters.copy()

    magnets = magnet_builder(p)
    refresh_matrices(magnets)
    lattice = build_lattice(ring_names, magnets)

    data = linear_optics(
        lattice,
        energy=p[energy_parameter],
        repetitions=repetitions,
        step=step,
    )

    correction = None

    if correct_chromatic:
        S1, S2, corrected_x, corrected_y = correct_chromaticity_from_data(
            lattice,
            data,
            family1=family1,
            family2=family2,
            target_x=target_chrom_x,
            target_y=target_chrom_y,
            repetitions=repetitions,
            step=step,
        )

        correction = [family1, S1, family2, S2, corrected_x, corrected_y]
        data[_linear_data_index("CHROM_X")] = corrected_x
        data[_linear_data_index("CHROM_Y")] = corrected_y

        for family, strength in ((family1, S1), (family2, S2)):
            parameter_name = correction_parameter_map.get(family)
            if parameter_name is not None:
                p[parameter_name] = strength

    return magnets, lattice, data, correction, p


def update_linear( #Updating the linear data when changed some varialbes
    lattice,
    data,
    parameters,
    edited_variables,
    correct_chromatic,
    family1,
    family2,
    target_chrom_x,
    target_chrom_y,
    repetitions,
    step,
    linear_variables,
    chromatic_variables,
    parameter_map,
    magnet_builder,
    correction_parameter_map,
    energy_parameter,
):
    """Update edited parameters and repeat only the affected calculations."""
    p = parameters.copy()
    data = list(data)
    edited = set(edited_variables)
    linear_variables = set(linear_variables)
    chromatic_variables = set(chromatic_variables)

    known_variables = set(parameter_map) | linear_variables | chromatic_variables
    unknown = edited - known_variables
    if unknown:
        raise KeyError("Unknown parameters: " + str(sorted(unknown)))

    new_magnets = {}
    if any(parameter_map.get(variable) for variable in edited):
        new_magnets = {elem[0]: elem for elem in magnet_builder(p)}

    changed_magnets = set()
    for variable in edited:
        for magnet_name, field in parameter_map.get(variable, []):
            field_index = _magnet_field_index(field)
            elem = get_magnet(lattice, magnet_name)
            elem[field_index] = new_magnets[magnet_name][field_index]
            if str(field).upper() in ("LENGTH", "ANGLE", "K"):
                changed_magnets.add(magnet_name)

    for magnet_name in changed_magnets:
        compute_matrices(get_magnet(lattice, magnet_name))

    linear_changed = bool(edited & linear_variables)
    chromatic_changed = bool(edited & chromatic_variables)

    if linear_changed:
        data = linear_optics(
            lattice,
            energy=p[energy_parameter],
            repetitions=repetitions,
            step=step,
        )

    correction = None
    if correct_chromatic and (linear_changed or chromatic_changed):
        S1, S2, corrected_x, corrected_y = correct_chromaticity_from_data(
            lattice,
            data,
            family1=family1,
            family2=family2,
            target_x=target_chrom_x,
            target_y=target_chrom_y,
            repetitions=repetitions,
            step=step,
        )
        correction = [family1, S1, family2, S2, corrected_x, corrected_y]
        data[_linear_data_index("CHROM_X")] = corrected_x
        data[_linear_data_index("CHROM_Y")] = corrected_y

        for family, strength in ((family1, S1), (family2, S2)):
            parameter_name = correction_parameter_map.get(family)
            if parameter_name is not None:
                p[parameter_name] = strength
    elif chromatic_changed and not linear_changed:
        chrom_x, chrom_y = chromaticity_with_sextupoles(
            lattice, repetitions=repetitions, step=step
        )
        data[_linear_data_index("CHROM_X")] = chrom_x
        data[_linear_data_index("CHROM_Y")] = chrom_y

    return lattice, data, correction, p

# OPTIONAL TESTING, REPORTING, AND PLOTTING
def check_linear_lattice(lattice, data):
    """Return basic consistency errors for the current linear lattice."""
    M4 = data[2]

    J2 = np.array([[0.0, 1.0], [-1.0, 0.0]])
    J4 = np.zeros((4, 4))
    J4[:2, :2] = J2
    J4[2:, 2:] = J2

    symplectic_error = np.linalg.norm(M4.T @ J4 @ M4 - J4)

    cs_values = data[11]
    disp_values = data[12]

    twiss_closure = np.max(np.abs(cs_values[-1] - cs_values[0]))
    dispersion_closure = np.max(np.abs(disp_values[-1] - disp_values[0]))

    cs_identity_x = np.max(
        np.abs(cs_values[:, 0] * cs_values[:, 2] - cs_values[:, 1]**2 - 1.0)
    )
    cs_identity_y = np.max(
        np.abs(cs_values[:, 3] * cs_values[:, 5] - cs_values[:, 4]**2 - 1.0)
    )

    bending_ring = sum(elem[3] for elem in lattice if elem[1] == "bending")

    return [
        symplectic_error,
        twiss_closure,
        dispersion_closure,
        cs_identity_x,
        cs_identity_y,
        bending_ring,
    ]
