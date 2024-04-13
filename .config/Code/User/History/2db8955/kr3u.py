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
    x0 = x[0], xf = x[-1]

    prog.AddLinearEqualityConstraint(x0, initial_state)

    
    