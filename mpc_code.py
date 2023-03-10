from time import time
import casadi as ca
import numpy as np
from casadi import sin, cos, pi
from simulation_code import simulate

# setting matrix_weights' variables
Q_x = 5
Q_y = 5
Q_theta = 0.0
R1 = 0.0
R2 = 10

step_horizon = 0.1  # time between steps in seconds
N = 10            # number of look ahead steps
rob_diam = 0.3      # diameter of the robot
wheel_radius = 1    # wheel radius
Lx = 0.3            # L in J Matrix (half robot x-axis length)
Ly = 0.3            # l in J Matrix (half robot y-axis length)
sim_time = 100     # simulation time

# specs
x_init = 0
y_init = 0
theta_init = 0
x_target = 10
y_target = 15
theta_target = 0.0

v_max = 1.0
v_min = 0
w_max = 0.3
w_min = -0.3


def shift_timestep(step_horizon, t0, state_init, u, f):
    f_value = f(state_init, u[:, 0])
    next_state = ca.DM.full(state_init + (step_horizon * f_value))

    t0 = t0 + step_horizon
    u0 = ca.horzcat(
        u[:, 1:],
        ca.reshape(u[:, -1], -1, 1)
    )

    return t0, next_state, u0


def DM2Arr(dm):
    return np.array(dm.full())


# state symbolic variables
x = ca.SX.sym('x')
y = ca.SX.sym('y')
theta = ca.SX.sym('theta')
states = ca.vertcat(
    x,
    y,
    theta
)
n_states = states.numel()

# 3
v_ = ca.SX.sym('v_')
w_ = ca.SX.sym('w_')
controls = ca.vertcat(v_, w_)
n_controls = controls.numel()

# matrix containing all states over all time steps +1 (each column is a state vector)
X = ca.SX.sym('X', n_states, N + 1)

# matrix containing all control actions over all time steps (each column is an action vector)
U = ca.SX.sym('U', n_controls, N)

# coloumn vector for storing initial state and target state
P = ca.SX.sym('P', n_states + n_states)

# state weights matrix (Q_X, Q_Y, Q_THETA)
Q = ca.diagcat(Q_x, Q_y, Q_theta)

###############
R = ca.diagcat(R1, R2)
RHS = ca.vertcat(
    v_*cos(theta) - wheel_radius*w_*sin(theta),
    v_*sin(theta) + wheel_radius*w_*cos(theta),
    w_
)

# maps controls from [va, vb, vc, vd].T to [vx, vy, omega].T
f = ca.Function('f', [states, controls], [RHS])


cost_fn = 0  # cost function
g = X[:, 0] - P[:n_states]  # constraints in the equation


# runge kutta
for k in range(N):
    st = X[:, k]
    con = U[:, k]
    cost_fn = cost_fn \
        + (st - P[n_states:]).T @ Q @ (st - P[n_states:]) \
        + con.T @ R @ con \
        ##+ CM(st)
    st_next = X[:, k+1]

    st_next_shift = st + f(st, con)*step_horizon
    g = ca.vertcat(g, st_next - st_next_shift)


# # circle obstackle
# n_obs = 1
# obs_x = np.ones((n_obs, 1))*5
# obs_y = np.linspace(5, 15, n_obs)
# obs_diam = 2


# for k in range(N+1):
#     for j in range(n_obs):
#         g = ca.vertcat(g, -ca.sqrt((X[0, k]-obs_x[j])**2 +
#                        (X[1, k]-obs_y[j])**2) + rob_diam/2 + obs_diam/2)


# ellipse obstackle
n_obs = 1
center = [5, 5]
scale = [5+rob_diam/2, 0.1+rob_diam/2]  
rot = 3*pi/4
cR = cos(rot)
sR = sin(rot)

for k in range(N+1):
    for j in range(n_obs):
        g = ca.vertcat(g, 1 - (((X[0,k]-center[0])*cR+(X[1,k]-center[1])*sR)/scale[0])**2 - (((X[0,k]-center[0])*sR-(X[1,k]-center[1])*cR)/scale[1])**2)


OPT_variables = ca.vertcat(
    X.reshape((-1, 1)),   # Example: 3x11 ---> 33x1 where 3=states, 11=N+1
    U.reshape((-1, 1))
)
nlp_prob = {
    'f': cost_fn,
    'x': OPT_variables,
    'g': g,
    'p': P
}

opts = {
    'ipopt': {
        'max_iter': 2000,
        'print_level': 0,
        'acceptable_tol': 1e-8,
        'acceptable_obj_change_tol': 1e-6
    },
    'print_time': 0
}

solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

lbx = ca.DM.zeros((n_states*(N+1) + n_controls*N, 1))
ubx = ca.DM.zeros((n_states*(N+1) + n_controls*N, 1))

lbx[0: n_states*(N+1): n_states] = -ca.inf     # X lower bound
lbx[1: n_states*(N+1): n_states] = -ca.inf     # Y lower bound
lbx[2: n_states*(N+1): n_states] = -ca.inf     # theta lower bound

ubx[0: n_states*(N+1): n_states] = ca.inf      # X upper bound
ubx[1: n_states*(N+1): n_states] = ca.inf      # Y upper bound
ubx[2: n_states*(N+1): n_states] = ca.inf      # theta upper bound

lbg = ca.DM.zeros((n_states*(N+1), 1))          # equal constraints
ubg = ca.DM.zeros((n_states*(N+1), 1))          # equal constraints

lbg = ca.vertcat(lbg, -ca.inf*ca.DM.ones(n_obs*(N+1), 1)
                 )      # inequal constraints
ubg = ca.vertcat(ubg, ca.DM.zeros(n_obs*(N+1), 1))      # inequal constraints

k = n_states*(N+1)
while k < n_states*(N+1) + n_controls*N:
    lbx[k] = v_min
    lbx[k+1] = w_min
    ubx[k] = v_max
    ubx[k+1] = w_max
    k += 2


args = {
    'lbg': lbg,
    'ubg': ubg,
    'lbx': lbx,
    'ubx': ubx
}

t0 = 0
state_init = ca.DM([x_init, y_init, theta_init])        # initial state
state_target = ca.DM([x_target, y_target, theta_target])  # target state

# xx = DM(state_init)
t = ca.DM(t0)

u0 = ca.DM.zeros((n_controls, N))  # initial control
X0 = ca.repmat(state_init, 1, N+1)         # initial state full


mpc_iter = 0
cat_states = DM2Arr(X0)
cat_controls = DM2Arr(u0[:, 0])
times = np.array([[0]])


###############################################################################

if __name__ == '__main__':
    main_loop = time()  # return time in sec
    while (ca.norm_2(state_init - state_target) > 1e-1) and (mpc_iter * step_horizon < sim_time):
        t1 = time()
        args['p'] = ca.vertcat(
            state_init,    # current state
            state_target   # target state
        )
        # optimization variable current state
        args['x0'] = ca.vertcat(
            ca.reshape(X0, n_states*(N+1), 1),
            ca.reshape(u0, n_controls*N, 1)
        )

        sol = solver(
            x0=args['x0'],
            lbx=args['lbx'],
            ubx=args['ubx'],
            lbg=args['lbg'],
            ubg=args['ubg'],
            p=args['p']
        )

        u = ca.reshape(sol['x'][n_states * (N + 1):], n_controls, N)
        X0 = ca.reshape(sol['x'][: n_states * (N+1)], n_states, N+1)

        cat_states = np.dstack((
            cat_states,
            DM2Arr(X0)
        ))

        cat_controls = np.vstack((
            cat_controls,
            DM2Arr(u[:, 0])
        ))
        t = np.vstack((
            t,
            t0
        ))

        t0, state_init, u0 = shift_timestep(step_horizon, t0, state_init, u, f)

        # print(X0)
        X0 = ca.horzcat(
            X0[:, 1:],
            ca.reshape(X0[:, -1], -1, 1)
        )

        # xx ...
        t2 = time()
        # print(mpc_iter)
        # print(t2-t1)
        times = np.vstack((
            times,
            t2-t1
        ))

        mpc_iter = mpc_iter + 1

    main_loop_time = time()
    ss_error = ca.norm_2(state_init[:2] - state_target[:2])

    print('\n\n')
    print('Total time: ', main_loop_time - main_loop)
    print('avg iteration time: ', np.array(times).mean() * 1000, 'ms')
    print('final error: ', ss_error)

    # simulate
    simulate(cat_states, cat_controls, times, step_horizon, N,
             np.array([x_init, y_init, theta_init, x_target, y_target, theta_target]), save=False)
