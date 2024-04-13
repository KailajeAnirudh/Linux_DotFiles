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
# Create a MultibodyPlant for the arm
file_name = "/home/dhruv/Hop-Skip-and-Jump/models/planar_walker.urdf"

builder = DiagramBuilder()
meshcat = StartMeshcat()
