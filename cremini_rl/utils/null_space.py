import numpy as np
import torch
from scipy.linalg import qr, svd

def batch_qr_null_tensor(A, tol=None):
    Q, R = torch.linalg.qr(A.transpose(-1, -2), mode='complete')
    tol = torch.amax(A, dim=(-2, -1)) * torch.finfo(R.dtype).eps if tol is None else tol
    while tol.dim() < R.dim() - 1:
        tol = tol.unsqueeze(-1)
    rnk = torch.full(A.shape[:-2], min(A.shape[-2:])).to(device=A.device) - torch.searchsorted(torch.abs(torch.diagonal(R, dim1=-2, dim2=-1)).flip(dims=(-1,)), tol)
    return Q[..., rnk.min():].conj()

def batch_smooth_basis_tensor(A, T0=None):
    """
    Compute the null space matrix suggested by:
    On the computation of multidimensional solution manifolds of parametrized equations
    """
    Ux = batch_qr_null_tensor(A)
    if T0 is None:
        T0 = torch.zeros(Ux.shape[-2:], device=Ux.device, dtype=Ux.dtype)
        T0.fill_diagonal_(1.0)

    else:
        assert T0.shape == (Ux.shape[-2], Ux.shape[-1])

    U0 = Ux.transpose(-1, -2) @ T0
    U, s, Vh = torch.linalg.svd(U0)
    Q = U @ Vh
    return (Ux @ Q)

def batch_smooth_basis(A, T0=None):
    return np.array(batch_smooth_basis_tensor(torch.tensor(A).double(), T0), float)

def qr_null(A, tol=None):
    Q, R = qr(A.T, mode='full')
    tol = np.max(A) * np.finfo(R.dtype).eps if tol is None else tol
    rnk = min(A.shape) - np.abs(np.diag(R))[::-1].searchsorted(tol)
    return Q[:, rnk:].conj()


def smooth_basis(A, T0=None):
    """
    Compute the null space matrix suggested by:
    On the computation of multidimensional solution manifolds of parametrized equations
    """
    Ux = qr_null(A)  # [[ 0.40824829], [-0.81649658], [ 0.40824829]]
    if T0 is None:
        T0 = np.zeros(Ux.shape)
        np.fill_diagonal(T0, 1.0)
    else:
        assert T0.shape == (Ux.shape[0], Ux.shape[0] - Ux.shape[1])

    U0 = Ux.T @ T0  # [[0.40824829]]
    U, s, Vh = svd(U0)  # [[1.]] [0.40824829] [[1.]]
    Q = U @ Vh  # [[1.]]
    return Ux @ Q


if __name__ == "__main__":
    test_arr = np.arange(3).reshape(1, 3)
    print(test_arr)
    res1 = smooth_basis(test_arr)

    test_arr = np.array([test_arr, test_arr, 2 * test_arr, 3 * test_arr])

    print(test_arr.shape)
    print(test_arr)

    res2 = batch_smooth_basis(test_arr)

    print(res1, res2)
