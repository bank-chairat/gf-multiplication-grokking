import galois
import numpy as np


def build_gf_mult_dataset(m=4):
    """Build the full multiplication table for GF(2^m).

    Returns:
        a: Array of first operands.
        b: Array of second operands.
        products: Array of products a * b in GF(2^m).
        GF: The Galois field class used for the computation.
    """
    GF = galois.GF(2**m)
    n = 2**m

    a_vals = []
    b_vals = []
    products = []

    for a in range(n):
        for b in range(n):
            product = int(GF(a) * GF(b))

            a_vals.append(a)
            b_vals.append(b)
            products.append(product)

    return (
        np.array(a_vals),
        np.array(b_vals),
        np.array(products),
        GF,
    )
