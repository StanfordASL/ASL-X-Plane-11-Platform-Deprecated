from functools import partial

from jfi import jaxm

def simulate(A, B, d, L, l, x, T):
    xs, us = [x], []
    for i in range(T):
        u = L @ x + l
        x = A @ x + B @ u + d
        xs.append(x)
        us.append(u)
    return jaxm.stack(xs, 0), jaxm.stack(us, 0)


bmv = lambda A, x: (A @ x[..., None])[..., 0]


def _lqr_loop(A, B, d, Q, R, q, r, carry, x):
    V, v = carry
    Su = r + bmv(B.T, v).T + bmv((V @ B).T, d).T
    Suu = R + B.T @ V @ B
    Sux = B.T @ V @ A
    L, l = -jaxm.linalg.solve(Suu, Sux), -jaxm.linalg.solve(Suu, Su)
    v = q + A.T @ (v + V @ d) + Sux.T @ l
    V = Q + A.T @ V @ A - L.T @ Suu @ L
    return (V, v), (L, l)


def LQR(A, B, d, Q, R, q, r, T=20):
    V, v = Q, q
    return jaxm.lax.scan(
        partial(_lqr_loop, A, B, d, Q, R, q, r), (V, v), xs=None, length=T, reverse=True
    )[1]


@partial(jaxm.jit, static_argnames="T")
def design_LQR_controller(A, B, d, Q, R, x_ref, u_ref, T=20):
    q, r = -bmv(Q, x_ref), -bmv(R, u_ref)
    Ls, ls = LQR(A, B, d, Q, R, q, r, T=T)
    return Ls[0], ls[0]