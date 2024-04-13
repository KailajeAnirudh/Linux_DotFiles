import matplotlib.pyplot as plt
import numpy as np
import importlib
from CollocationConstraits import *

from pydrake.all import (
    DiagramBuilder, Simulator, FindResourceOrThrow, MultibodyPlant, PiecewisePolynomial, SceneGraph,
    Parser, JointActuatorIndex, MathematicalProgram, Solve
)

def find_trajectory (num_knot_points, initial_state, jumpheight, jump_height_tol, jump_time):
    builder = DiagramBuilder()
    plant_f = builder.AddSystem(MultibodyPlant(0.0))
    (planar_walker,) = Parser(plant_f).AddModels("/home/anirudhkailaje/Documents/01_UPenn/02_MEAM5170/05_Madness/src/planar_walker.urdf")
    W = plant_f.world_frame()
    base = plant_f.GetFrameByName("base", planar_walker)
    Left_foot = plant_f.GetFrameByName("left_lower_leg", planar_walker)
    Right_foot = plant_f.GetFrameByName("right_lower_leg", planar_walker)

    plant_f.WeldFrames(W, base)
    plant_f.Finalize()

    context_f = plant_f.CreateDefaultContext()
    # Create AutoDiffXd plant and corresponding context.
    plant_ad = plant_f.ToAutoDiffXd() #Autodiff plant
    context_ad = plant_ad.CreateDefaultContext()

    n_q = plant_ad.num_positions(); n_v = plant_ad.num_velocities(); n_x = n_q+n_v; n_u = plant_ad.num_actuators()

    prog = MathematicalProgram()
    x = np.array([prog.NewContinuousVariables(n_x, f"x_{i}") for i in range(num_knot_points)], dtype="object")
    u = np.array([prog.NewContinuousVariables(n_u, f"u_{i}") for i in range(num_knot_points)], dtype="object")
    lambda_c = np.array([prog.NewContinuousVariables(6, f"lambda_{i}") for i in range(num_knot_points)], dtype="object")
    lambda_col = np.array([prog.NewContinuousVariables(6, f"collocationlambda_{i}") for i in range(num_knot_points-1)], dtype="object")
    timesteps = np.linspace(0.0, jump_time, num_knot_points)
    x0 = x[0]; xf = x[-1]

    prog.AddLinearEqualityConstraint(x0, initial_state)
    
    AddCollocationConstraints(prog, plant_ad, context_ad, num_knot_points, x, u, lambda_c, lambda_col, timesteps)

    for i in range(N-1):
       prog.AddQuadraticCost(0.5*(timesteps[i+1]-timesteps[i])*((u[i].T@u[i])+(u[i+1].T@u[i+1])))
    
    x_init = x_init = np.zeros((num_knot_points, n_x))
    u_init = np.zeros((num_knot_points, n_u))

    prog.SetInitialGuess(x, x_init)
    prog.SetInitialGuess(u, u_init)

    result = Solve(prog)
    x_sol = result.GetSolution(x)
    u_sol = result.GetSolution(u)
    print('optimal cost: ', result.get_optimal_cost())
    print('x_sol: ', x_sol)
    print('u_sol: ', u_sol)
    print("Done")

if __name__ == '__main__':
    N = 5
    q = np.zeros(14); q[0] = 0; q[1] = 0.8
    theta = -np.arccos(q[1])
    q[3] = theta; q[4] = -2 * theta
    q[5] = theta;   q[6] = -2 * theta
    initial_state =q
    
    x_traj, u_traj, prog, _, _ = find_trajectory(N, initial_state, jumpheight=0.25, jump_height_tol = 1e-2, jump_time=2)