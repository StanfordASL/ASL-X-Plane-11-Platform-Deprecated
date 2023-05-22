from copy import copy

from jfi import jaxm

bmv = lambda A, x: (A @ x[..., None])[..., 0]


def dynamics(state, control, params):
    """Simple dynamics of an airplane."""
    x, y, z, v, vh, pitch, roll, yaw, dpitch, droll, dyaw = state

    # position
    xp = (v * jaxm.cos(yaw + 0 * params["heading_correction"])).reshape(())
    yp = (v * jaxm.sin(yaw + 0 * params["heading_correction"])).reshape(())
    dt = params["dt_sqrt"] ** 2

    dynamic_states = jaxm.stack([v, vh, pitch, roll, dpitch, droll, dyaw])
    statep_partial = bmv(params["Wx"], dynamic_states) + bmv(params["Wu"], control) + params["b"]

    statep = jaxm.cat([jaxm.stack([xp, yp]), statep_partial])
    return state + dt * statep


params0 = {
    "dt_sqrt": jaxm.sqrt(jaxm.array([0.5])),
    "heading_correction": jaxm.array([0.0]),
    "Wx": jaxm.randn((9, 7)),
    "Wu": jaxm.randn((9, 4)),
    "b": jaxm.randn(9),
}


@jaxm.jit
def fwd_fn(state, control, params):
    return jaxm.vmap(dynamics, in_axes=(0, 0, None))(state, control, params)


@jaxm.jit
def f_fx_fu_fn(X, U, params):
    bshape = X.shape[:-1]
    X, U = X.reshape((-1, X.shape[-1])), U.reshape((-1, U.shape[-1]))
    f = fwd_fn(X, U, params)
    fx = jaxm.vmap(
        lambda x, u: jaxm.jacobian(
            lambda x, u: fwd_fn(x[None, ...], u[None, ...], params)[0, ...], argnums=0
        )(x, u)
    )(X, U)
    fu = jaxm.vmap(
        lambda x, u: jaxm.jacobian(
            lambda x, u: fwd_fn(x[None, ...], u[None, ...], params)[0, ...], argnums=1
        )(x, u)
    )(X, U)
    fsh, fxsh, fush = bshape + f.shape[-1:], bshape + fx.shape[-2:], bshape + fu.shape[-2:]
    return f.reshape(fsh), fx.reshape(fxsh), fu.reshape(fush)


####################################################################################################


def dynamics2(state, control, params):
    """Simple dynamics of an airplane."""
    #state1, yaw_s, yaw_c, state2 = state[:7], state[7], state[8], state[9:]
    #yaw = jaxm.arctan2(yaw_s, yaw_c)
    #state = jaxm.cat([state1, jaxm.array([yaw]), state2])

    state = dynamics(state, control, params)
    #state1, yaw, state2 = state[:7], state[7], state[8:]
    #state = jaxm.cat([state1, jaxm.array([jaxm.sin(yaw), jaxm.cos(yaw)]), state2])
    return state


@jaxm.jit
def fwd_fn2(state, control, params):
    return jaxm.vmap(dynamics2, in_axes=(0, 0, None))(state, control, params)


@jaxm.jit
def f_fx_fu_fn2(X, U, params):
    bshape = X.shape[:-1]
    X, U = X.reshape((-1, X.shape[-1])), U.reshape((-1, U.shape[-1]))
    f = fwd_fn2(X, U, params)
    fx = jaxm.vmap(
        lambda x, u: jaxm.jacobian(
            lambda x, u: fwd_fn2(x[None, ...], u[None, ...], params)[0, ...], argnums=0
        )(x, u)
    )(X, U)
    fu = jaxm.vmap(
        lambda x, u: jaxm.jacobian(
            lambda x, u: fwd_fn2(x[None, ...], u[None, ...], params)[0, ...], argnums=1
        )(x, u)
    )(X, U)
    fsh, fxsh, fush = bshape + f.shape[-1:], bshape + fx.shape[-2:], bshape + fu.shape[-2:]
    return f.reshape(fsh), fx.reshape(fxsh), fu.reshape(fush)


####################################################################################################


def int_dynamics2(state, control, params):
    dt = params["dt_sqrt"] ** 2
    aero_state = state[:11]
    next_aero_state = dynamics2(aero_state, control, params)
    pos_int = state[11:14]
    ang_int = state[14:17]
    pos_int = pos_int + dt * (next_aero_state[:3] - params["pos_ref"])
    ang_int = ang_int + dt * (next_aero_state[5:8] - params["ang_ref"])
    return jaxm.cat([next_aero_state, pos_int, ang_int])


def int_fwd_fn2(state, control, params):
    return jaxm.vmap(int_dynamics2, in_axes=(0, 0, None))(state, control, params)


@jaxm.jit
def int_f_fx_fu_fn2(X, U, params):
    bshape = X.shape[:-1]
    X, U = X.reshape((-1, X.shape[-1])), U.reshape((-1, U.shape[-1]))
    f = int_fwd_fn2(X, U, params)
    fx = jaxm.vmap(
        lambda x, u: jaxm.jacobian(
            lambda x, u: int_fwd_fn2(x[None, ...], u[None, ...], params)[0, ...], argnums=0
        )(x, u)
    )(X, U)
    fu = jaxm.vmap(
        lambda x, u: jaxm.jacobian(
            lambda x, u: int_fwd_fn2(x[None, ...], u[None, ...], params)[0, ...], argnums=1
        )(x, u)
    )(X, U)
    fsh, fxsh, fush = bshape + f.shape[-1:], bshape + fx.shape[-2:], bshape + fu.shape[-2:]
    return f.reshape(fsh), fx.reshape(fxsh), fu.reshape(fush)


####################################################################################################


def nn_dynamics(state, control, params):
    """Simple dynamics of an airplane."""
    x, y, z, v, vh, pitch, roll, yaw, dpitch, droll, dyaw = state

    # position
    xp = (v * jaxm.cos(yaw + params["heading_correction"])).reshape(())
    yp = (v * jaxm.sin(yaw + params["heading_correction"])).reshape(())
    dt = params["dt_sqrt"] ** 2

    dt = params["dt_sqrt"] ** 2

    dynamic_states = jaxm.stack([v, vh, pitch, roll, dpitch, droll, dyaw])

    Z = dynamic_states
    for i in range(3):
        Z = params[f"Wx{i}"] @ Z + params[f"b{i}"] + params[f"Wu{i}"] @ control
        if i < 3 - 1:
            Z = jaxm.tanh(Z)
    return state + dt * jaxm.cat([jaxm.stack([xp, yp]), Z])


@jaxm.jit
def nn_fwd_fn(state, control, params):
    return jaxm.vmap(nn_dynamics, in_axes=(0, 0, None))(state, control, params)


@jaxm.jit
def nn_f_fx_fu_fn(X, U, params):
    if X.ndim == 3:
        return jaxm.vmap(nn_f_fx_fu_fn, in_axes=(0, 0, None))(X, U, params)
    f = nn_fwd_fn(X, U, params)
    fx = jaxm.vmap(
        lambda x, u: jaxm.jacobian(
            lambda x, u: nn_fwd_fn(x[None, ...], u[None, ...], params)[0, ...], argnums=0
        )(x, u)
    )(X, U)
    fu = jaxm.vmap(
        lambda x, u: jaxm.jacobian(
            lambda x, u: nn_fwd_fn(x[None, ...], u[None, ...], params)[0, ...], argnums=1
        )(x, u)
    )(X, U)
    return f, fx, fu


nn_params0 = {
    "Wx0": jaxm.randn((32, 7)),
    "Wu0": jaxm.randn((32, 4)),
    "b0": jaxm.randn(32),
    "Wx1": jaxm.randn((32, 32)),
    "Wu1": jaxm.randn((32, 4)),
    "b1": jaxm.randn(32),
    "Wx2": jaxm.randn((9, 32)),
    "Wu2": jaxm.randn((9, 4)),
    "b2": jaxm.randn(9),
    "heading_correction": jaxm.array([0.0]),
    "dt_sqrt": jaxm.sqrt(jaxm.array([0.5])),
}

####################################################################################################


def aero_dynamics_delta(state, control, params):
    # Unpack state vector (position, Euler angles, velocity, angular rates)
    v, vh = state[3:5]
    pitch, roll, yaw = state[5:8]
    dpitch, droll, dyaw = state[8:11]
    vel_body = jaxm.array([v, 0.0, vh])
    psi, theta, phi = yaw, pitch, roll
    euler = jaxm.array([roll, pitch, yaw])
    omega = jaxm.array([droll, dpitch, dyaw])

    # Unpack control inputs and scale them
    u_scale = params["control_scaling"]
    elevator, aileron, rudder, throttle = control * u_scale

    # Unpack aircraft parameters
    (
        mass,
        Jx,
        Jy,
        Jz,
        Jxz,
        g,
        S,
        b,
        c,
        rho,
        CD0,
        CL0,
        CDalpha,
        CLalpha,
        CDq,
        CLq,
        CDdeltae,
        CLdeltae,
    ) = params["aircraft_params"]
    # mass = 1000
    # Jx = Jx ** 2
    # Jy = Jy ** 2
    # Jz = Jz ** 2
    # g = 9.81
    # S = 16.2
    # b = 11.0
    ##c = 0.2
    # c = c ** 2 + 0.1
    # rho = 1.2
    ##CDalpha = CDalpha ** 2
    ##CLalpha = CLalpha ** 2
    ##CDq = CDq ** 2
    ##CLq = CLq ** 2
    ##CDdeltae = CDdeltae ** 2
    ##CLdeltae = CLdeltae ** 2

    # Pre-calculate trigonometric functions
    sin_phi = jaxm.sin(phi)
    cos_phi = jaxm.cos(phi)
    sin_theta = jaxm.sin(theta)
    cos_theta = jaxm.cos(theta)
    sin_psi = jaxm.sin(psi)
    cos_psi = jaxm.cos(psi)

    # Rotation matrix from body to ground coordinates
    R = jaxm.array(
        [
            [
                cos_psi * cos_theta,
                cos_psi * sin_theta * sin_phi - sin_psi * cos_phi,
                cos_psi * sin_theta * cos_phi + sin_psi * sin_phi,
            ],
            [
                sin_psi * cos_theta,
                sin_psi * sin_theta * sin_phi + cos_psi * cos_phi,
                sin_psi * sin_theta * cos_phi - cos_psi * sin_phi,
            ],
            [-sin_theta, cos_theta * sin_phi, cos_theta * cos_phi],
        ]
    )

    # Convert body velocities to ground velocities
    vel = jaxm.dot(R, vel_body)

    # Calculate aerodynamic forces and moments
    qS = 0.5 * rho * jaxm.linalg.norm(vel_body) ** 2 * S
    CL = (
        CL0
        + CLalpha * euler[1]
        + CLq * (0.5 * c * omega[2] / jaxm.linalg.norm(vel_body))
        + CLdeltae * elevator
    )
    CD = (
        CD0
        + CDalpha * euler[1]
        + CDq * (0.5 * c * omega[2] / jaxm.linalg.norm(vel_body))
        + CDdeltae * elevator
    )
    Lift = CL * qS
    Drag = CD * qS

    F_aero = jaxm.stack([throttle - Drag, 0, -Lift])  # Aerodynamic forces in body coordinates
    M_aero = jaxm.stack([aileron, elevator, rudder])  # Aerodynamic moments

    # Calculate gravitational force in body coordinates
    F_grav_body = jaxm.dot(R.T, jaxm.array([0, 0, mass * g]))

    # Combine forces and moments
    F = F_aero + F_grav_body
    M = M_aero

    # Calculate the derivative of the state vector
    dvel = F / mass
    ddpsi, ddtheta, ddphi = (
        jaxm.linalg.inv(jaxm.array([[Jx, 0, -Jxz], [0, Jy, 0], [-Jxz, 0, Jz]])) @ M
    )
    ddpitch, ddroll, ddyaw = ddtheta, ddphi, ddpsi

    dstate = jaxm.cat(
        [
            vel,
            jaxm.array([dvel[0], dvel[2]]),
            jaxm.array([dpitch, droll, dyaw]),
            jaxm.array([ddpitch, ddroll, ddyaw]),
        ]
    )
    return dstate


def aero_dynamics(state, control, params):
    state1, yaw_s, yaw_c, state2 = state[:7], state[7], state[8], state[9:]
    yaw = jaxm.arctan2(yaw_s, yaw_c)
    state = jaxm.cat([state1, jaxm.array([yaw]), state2])

    dt = params["dt_sqrt"] ** 2
    N = 5
    for _ in range(N):
        state += dt / N * aero_dynamics_delta(state, control, params)
    state1, yaw, state2 = state[:7], state[7], state[8:]
    state = jaxm.cat([state1, jaxm.array([jaxm.sin(yaw), jaxm.cos(yaw)]), state2])
    return state


def aero_fwd_fn(state, control, params):
    return jaxm.vmap(aero_dynamics, in_axes=(0, 0, None))(state, control, params)


@jaxm.jit
def aero_f_fx_fu_fn(X, U, params):
    if X.ndim == 3:
        return jaxm.vmap(aero_f_fx_fu_fn, in_axes=(0, 0, None))(X, U, params)
    f = aero_fwd_fn(X, U, params)
    fx = jaxm.vmap(
        lambda x, u: jaxm.jacobian(
            lambda x, u: aero_fwd_fn(x[None, ...], u[None, ...], params)[0, ...], argnums=0
        )(x, u)
    )(X, U)
    fu = jaxm.vmap(
        lambda x, u: jaxm.jacobian(
            lambda x, u: aero_fwd_fn(x[None, ...], u[None, ...], params)[0, ...], argnums=1
        )(x, u)
    )(X, U)
    return f, fx, fu


aero_params0 = {
    "control_scaling": 1e0 * jaxm.rand(4),
    "aircraft_params": 1e-3 * jaxm.rand(18),
    "dt_sqrt": jaxm.sqrt(jaxm.array([0.5])),
}


####################################################################################################


def int_aero_dynamics(state, control, params):
    dt = params["dt_sqrt"] ** 2
    aero_state = state[:11]
    next_aero_state = aero_dynamics(aero_state, control, params)
    pos_int = state[11:14]
    ang_int = state[14:17]
    pos_int = pos_int + dt * (next_aero_state[:3] - params["pos_ref"])
    ang_int = ang_int + dt * (next_aero_state[5:8] - params["ang_ref"])
    return jaxm.cat([next_aero_state, pos_int, ang_int])


def int_aero_fwd_fn(state, control, params):
    return jaxm.vmap(int_aero_dynamics, in_axes=(0, 0, None))(state, control, params)


@jaxm.jit
def int_aero_f_fx_fu_fn(X, U, params):
    if X.ndim == 3:
        return jaxm.vmap(int_aero_f_fx_fu_fn, in_axes=(0, 0, None))(X, U, params)
    f = int_aero_fwd_fn(X, U, params)
    fx = jaxm.vmap(
        lambda x, u: jaxm.jacobian(
            lambda x, u: int_aero_fwd_fn(x[None, ...], u[None, ...], params)[0, ...], argnums=0
        )(x, u)
    )(X, U)
    fu = jaxm.vmap(
        lambda x, u: jaxm.jacobian(
            lambda x, u: int_aero_fwd_fn(x[None, ...], u[None, ...], params)[0, ...], argnums=1
        )(x, u)
    )(X, U)
    return f, fx, fu


int_aero_params0 = dict(
    aero_params0,
    pos_ref=jaxm.randn(3),
    ang_ref=jaxm.randn(3),
)
