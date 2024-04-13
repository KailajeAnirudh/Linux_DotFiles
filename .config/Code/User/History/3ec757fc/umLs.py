from pydrake.all import FindResourceOrThrow, StartMeshcat, MeshcatVisualizer
import matplotlib.pyplot as plt
import numpy as np
import time
from pydrake.math import RigidTransform
from pydrake.all import (
    DiagramBuilder, Simulator, FindResourceOrThrow, MultibodyPlant, PiecewisePolynomial, SceneGraph,
    Parser, MultibodyPositionToGeometryPose, TrajectorySource, Demultiplexer, ConstantVectorSource
)
import importlib
from direct_col_dev import find_throwing_trajectory
from pydrake.all import AddMultibodyPlantSceneGraph, HalfSpace, CoulombFriction, MeshcatVisualizerParams, ConstantValueSource
from phase_switch import *
# Create a MultibodyPlant for the arm
meshcat = StartMeshcat()
file_name = "/home/dhruv/Hop-Skip-and-Jump/models/planar_walker.urdf"

builder = DiagramBuilder()
#### Designing our world ####
# Add a planar walker to the simulation
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.0005)
#Half space means a plane -> Ground Plane in particular
X_WG = HalfSpace.MakePose(np.array([0,0, 1]), np.zeros(3,))
plant.RegisterCollisionGeometry(plant.world_body(), X_WG, HalfSpace(), 
    "collision", CoulombFriction(1.0, 1.0))

parser = Parser(plant)
parser.AddModels(file_name)
plant.WeldFrames(plant.world_frame(),
    plant.GetBodyByName("base").body_frame(),
    RigidTransform.Identity())
plant.Finalize()

n_q = plant.num_positions()
n_v = plant.num_velocities()
n_u = plant.num_actuators()

N = 30
initial_state = np.zeros(14)
q = np.zeros((7,))
q[0] = 0; q[1] = 1
theta = -np.arccos(q[1])
q[3] = theta; q[4] = -2 * theta
q[5] = theta;   q[6] = -2 * theta
initial_state[:7] = q

final_state = initial_state
# final_configuration = np.array([np.pi, 0])
jump_height = 0.2
tf = 1/2
x_traj, u_traj, prog,  _, _ = find_throwing_trajectory(N, initial_state, jump_height, tf=tf, jumpheight_tol=5e-2)

PhaseSwitcher = PhaseSwitch(jump_height, tf, x_traj, 0.7)
OSC = 
