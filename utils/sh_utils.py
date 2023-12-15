import numpy as np
import torch
import torch.nn.functional as F

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]


def rotation_between_z(vec):
    """
    https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d/476311#476311
    Args:
        vec: [..., 3]

    Returns:
        R: [..., 3, 3]

    """
    v1 = -vec[..., 1]
    v2 = vec[..., 0]
    v3 = torch.zeros_like(v1)
    v11 = v1 * v1
    v22 = v2 * v2
    v33 = v3 * v3
    v12 = v1 * v2
    v13 = v1 * v3
    v23 = v2 * v3
    cos_p_1 = (vec[..., 2] + 1).clamp_min(1e-7)
    R = torch.zeros(vec.shape[:-1] + (3, 3,), dtype=torch.float32, device="cuda")
    R[..., 0, 0] = 1 + (-v33 - v22) / cos_p_1
    R[..., 0, 1] = -v3 + v12 / cos_p_1
    R[..., 0, 2] = v2 + v13 / cos_p_1
    R[..., 1, 0] = v3 + v12 / cos_p_1
    R[..., 1, 1] = 1 + (-v33 - v11) / cos_p_1
    R[..., 1, 2] = -v1 + v23 / cos_p_1
    R[..., 2, 0] = -v2 + v13 / cos_p_1
    R[..., 2, 1] = v1 + v23 / cos_p_1
    R[..., 2, 2] = 1 + (-v22 - v11) / cos_p_1
    R = torch.where((vec[..., 2] + 1 > 0)[..., None, None], R,
                    -torch.eye(3, dtype=torch.float32, device="cuda").expand_as(R))
    return R


def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert 4 >= deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    # if deg == 0 and result.shape[-1] == 1:
    #     result = result.expand_as(dirs)
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                  C1 * y * sh[..., 1] +
                  C1 * z * sh[..., 2] -
                  C1 * x * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                      C2[0] * xy * sh[..., 4] +
                      C2[1] * yz * sh[..., 5] +
                      C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                      C2[3] * xz * sh[..., 7] +
                      C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                          C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                          C3[1] * xy * z * sh[..., 10] +
                          C3[2] * y * (4 * zz - xx - yy) * sh[..., 11] +
                          C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                          C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                          C3[5] * z * (xx - yy) * sh[..., 14] +
                          C3[6] * x * (xx - 3 * yy) * sh[..., 15])

                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                              C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                              C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                              C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                              C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                              C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                              C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                              C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                              C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result


def eval_sh_coef(deg, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert 4 >= deg >= 0
    coeff = (deg + 1) ** 2
    results = torch.zeros(dirs.shape[:-1] + (coeff,), device=dirs.device)
    results[..., 0] = C0
    if deg > 0:
        x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
        results[..., 1] = -C1 * y
        results[..., 2] = C1 * z
        results[..., 3] = -C1 * x

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            results[..., 4] = C2[0] * xy
            results[..., 5] = C2[1] * yz
            results[..., 6] = C2[2] * (2.0 * zz - xx - yy)
            results[..., 7] = C2[3] * xz
            results[..., 8] = C2[4] * (xx - yy)

            if deg > 2:
                results[..., 9] = C3[0] * y * (3 * xx - yy)
                results[..., 10] = C3[1] * xy * z
                results[..., 11] = C3[2] * y * (4 * zz - xx - yy)
                results[..., 12] = C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
                results[..., 13] = C3[4] * x * (4 * zz - xx - yy)
                results[..., 14] = C3[5] * z * (xx - yy)
                results[..., 15] = C3[6] * x * (xx - 3 * yy)

                if deg > 3:
                    results[..., 16] = C4[0] * xy * (xx - yy)
                    results[..., 17] = C4[1] * yz * (3 * xx - yy)
                    results[..., 18] = C4[2] * xy * (7 * zz - 1)
                    results[..., 19] = C4[3] * yz * (7 * zz - 3)
                    results[..., 20] = C4[4] * (zz * (35 * zz - 30) + 3)
                    results[..., 21] = C4[5] * xz * (7 * zz - 3)
                    results[..., 22] = C4[6] * (xx - yy) * (7 * zz - 1)
                    results[..., 23] = C4[7] * xz * (xx - 3 * yy)
                    results[..., 24] = C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))

    return results


def RGB2SH(rgb):
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    return sh * C0 + 0.5
