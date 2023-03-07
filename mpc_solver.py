from time import time
import casadi as ca
import numpy as np
from casadi import sin, cos, pi

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

v_max = 1.0
v_min = 0
w_max = 0.3
w_min = -0.3

n_controls = 2
n_states = 3

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


def get_solver_args():
    # state symbolic variables
    x = ca.SX.sym('x')
    y = ca.SX.sym('y')
    theta = ca.SX.sym('theta')
    states = ca.vertcat(x,y,theta)
    n_states = states.numel()
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
            + con.T @ R @ con 
        st_next = X[:, k+1]

        st_next_shift = st + f(st, con)*step_horizon
        g = ca.vertcat(g, st_next - st_next_shift)

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

    # solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

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
        'ubx': ubx,
        'nlp': nlp_prob,
        'opt': opts,
        'X': X
    }
    return args, f
    

def mpc_controller(pose_init, pose_target, args, solver):
    pose_init = ca.DM(pose_init)
    pose_target = ca.DM(pose_target)
    args['p'] = ca.vertcat(pose_init, pose_target)
    X0 = ca.repmat(pose_init, 1, N+1)
    u0 = ca.DM.zeros((2, N))
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
    
    return u, X0


class MPCController():
    def __init__(self,):
        self.args, self.f = get_solver_args()

    def step(self, pose_init, pose_target, ells=None):
        if ells:
            self.update_costmap(ells)

        pose_init = ca.DM(pose_init)
        pose_target = ca.DM(pose_target)
        self.args['p'] = ca.vertcat(pose_init, pose_target)
        X0 = ca.repmat(pose_init, 1, N+1)
        u0 = ca.DM.zeros((2, N))
        self.args['x0'] = ca.vertcat(
            ca.reshape(X0, n_states*(N+1), 1),
            ca.reshape(u0, n_controls*N, 1)
        )
        solver = ca.nlpsol('solver', 'ipopt', self.args['nlp'], self.args['opt'])
        sol = solver(
            x0=self.args['x0'],
            lbx=self.args['lbx'],
            ubx=self.args['ubx'],
            lbg=self.args['lbg'],
            ubg=self.args['ubg'],
            p=self.args['p']
        )
        u = ca.reshape(sol['x'][n_states * (N + 1):], n_controls, N)
        X0 = ca.reshape(sol['x'][: n_states * (N+1)], n_states, N+1)
    
        return u, X0

    def update_costmap(self, ells):
        n_obs = len(ells)
        for k in range(N+1):
            for j in range(n_obs):
                x,y,a,b,rot = ells[j]
                a += rob_diam/2
                b += rob_diam/2
                cR = cos(rot)
                sR = sin(rot)
                self.args['nlp']['g'] = ca.vertcat(self.args['nlp']['g'], 
                                        1 - (((self.args['X'][0,k]-x)*cR+(self.args['X'][1,k]-y)*sR)/a)**2 - (((self.args['X'][0,k]-x)*sR-(self.args['X'][1,k]-y)*cR)/b)**2)

        self.args['lbg'] = ca.vertcat(self.args['lbg'], -ca.inf*ca.DM.ones(n_obs*(N+1), 1))
        self.args['ubg'] = ca.vertcat(self.args['ubg'], ca.DM.zeros(n_obs*(N+1), 1))



if __name__ == '__main__':
    main_loop = time()  # return time in sec
    t0 = 0
    state_init = ca.DM(np.array([0.0,0.0,1.57]))        # initial state
    state_target = ca.DM(np.array([10.0, 10.0, 0.0]))   # target state

    t = ca.DM(t0)
    u0 = ca.DM.zeros((n_controls, N))           # initial control
    X0 = ca.repmat(state_init, 1, N+1)          # initial state full


    mpc_iter = 0
    cat_states = DM2Arr(X0)
    cat_controls = DM2Arr(u0[:, 0])
    times = np.array([[0]])

    mpc_con = MPCController()
    f = mpc_con.f
    while (ca.norm_2(state_init - state_target) > 1e-1) and (mpc_iter * step_horizon < sim_time):
        t1 = time()
        u, X0 = mpc_con.step(state_init, state_target)

        cat_states = np.dstack((
            cat_states,
            DM2Arr(X0)
        ))

        cat_controls = np.vstack((
            cat_controls,
            DM2Arr(u[:, 0])
        ))

        t0, state_init, u0 = shift_timestep(step_horizon, t0, state_init, u, f)

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
    from simulation_code import simulate

    simulate(cat_states, cat_controls, times, step_horizon, N,
             np.array([0,0,0,10,10,0]), save=False)