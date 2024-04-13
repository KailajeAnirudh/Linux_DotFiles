import matplotlib.pyplot as plt
import numpy as np
import importlib

from pydrake.all import (
    DiagramBuilder, Simulator, FindResourceOrThrow, MultibodyPlant, PiecewisePolynomial, SceneGraph,
    Parser, JointActuatorIndex, MathematicalProgram, Solve
)

def find_trajectory(num_knot_points. initial_state, jumpheight, jump_height_tol, jump_time):
    builder = DiagramBuilder()
    plant_f = builder.AddSystem(MultibodyPlant(0.0))
    (planar_walker,) = Parser(plant_f).AddModels("/home/anirudhkailaje/Documents/01_UPenn/02_MEAM5170/05_Madness/src/planar_walker.urdf")
    W = plant_f.world_frame()
    base = plant_f.GetFrameByName("base", planar_walker)
    Left_foot = plant_f.GetFrameByName("left_lower_leg", planar_walker)
    Right_foot = plant_f.GetFrameByName("right_lower_leg", planar_walker)

    plant_f.WeldFrames(W, base)
    plant_f.Finalize()

    