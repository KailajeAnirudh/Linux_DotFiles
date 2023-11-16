import numpy as np
import pydrake.math
from pydrake.autodiffutils import AutoDiffXd



def EvaluateDynamics(robot, context, x, u):
  # Computes the dynamics xdot = f(x,u)

  robot.SetPositionsAndVelocities(context, x)
  n_v = robot.num_velocities()

  M = robot.CalcMassMatrixViaInverseDynamics(context)
  B = robot.MakeActuationMatrix()
  g = robot.CalcGravityGeneralizedForces(context)
  C = robot.CalcBiasTerm(context)

  M_inv = np.zeros((n_v,n_v)) 
  if(x.dtype == AutoDiffXd):
    M_inv = pydrake.math.inv(M)
  else:
    M_inv = np.linalg.inv(M)
  v_dot = M_inv @ (B @ u + g - C)
  return np.hstack((x[-n_v:], v_dot))

def CollocationConstraintEvaluator(robot, context, dt, x_i, u_i, x_ip1, u_ip1):
  n_x = robot.num_positions() + robot.num_velocities()
  h_i = np.zeros(n_x,)
  # TODO: Add a dynamics constraint using x_i, u_i, x_ip1, u_ip1, dt
  # You should make use of the EvaluateDynamics() function to compute f(x,u)
  fi = EvaluateDynamics(robot, context, x_i, u_i)
  fi1 = EvaluateDynamics(robot, context, x_ip1, u_ip1)

  s_halfway = (x_i+x_ip1)*0.5 - 0.125*(dt)*(fi1-fi)
  sdot_halfway = 1.5*(x_ip1-x_i)/dt - 0.25*(fi+fi1)
  u_halfway = (u_i+u_ip1)*0.5

  h_i = sdot_halfway - EvaluateDynamics(robot, context, s_halfway, u_halfway)
  return h_i

def AddCollocationConstraints(prog, robot, context, N, x, u, timesteps):
  n_u = robot.num_actuators()
  n_x = robot.num_positions() + robot.num_velocities()
  
  def CollocationConstraintHelper(vars):
    x_i = vars[:n_x]
    u_i = vars[n_x:n_x + n_u]
    x_ip1 = vars[n_x + n_u: 2*n_x + n_u]
    u_ip1 = vars[-n_u:]
    return CollocationConstraintEvaluator(robot, context, timesteps[i+1] - timesteps[i], x_i, u_i, x_ip1, u_ip1)
  
  for i in range(N - 1):      
    # TODO: Within this loop add the dynamics constraints for segment i (aka collocation constraints)
    #       to prog
    # Hint: use prog.AddConstraint(CollocationConstraintHelper, lb, ub, vars)
    # where vars = hstack(x[i], u[i], ...)
    lower_bound = np.zeros(n_x)
    upper_bound = lower_bound
    prog.AddConstraint(CollocationConstraintHelper, lower_bound, upper_bound, np.hstack([x[i], u[i], x[i+1], u[i+1]]) )
