import numpy as np
from pydrake.multibody.all import JacobianWrtVariable
import pydrake.math
from pydrake.autodiffutils import AutoDiffXd


def EvaluateDynamics(planar_arm, context, x, u):
  # Computes the dynamics xdot = f(x,u)

  planar_arm.SetPositionsAndVelocities(context, x)
  n_v = planar_arm.num_velocities()

  M = planar_arm.CalcMassMatrixViaInverseDynamics(context)
  B = planar_arm.MakeActuationMatrix()
  g = planar_arm.CalcGravityGeneralizedForces(context)
  C = planar_arm.CalcBiasTerm(context)

  M_inv = np.zeros((n_v,n_v)) 
  if(x.dtype == AutoDiffXd):
    M_inv = pydrake.math.inv(M)
  else:
    M_inv = np.linalg.inv(M)
  v_dot = M_inv @ (B @ u + g - C)

  return np.hstack((x[-n_v:], v_dot))

def CollocationConstraintEvaluator(planar_arm, context, dt, x_i, u_i, x_ip1, u_ip1, lambda_c_i, lambda_c_ip1):
  n_x = planar_arm.num_positions() + planar_arm.num_velocities()
  h_i = np.zeros(n_x,)
  # TODO: Add a dynamics constraint using x_i, u_i, x_ip1, u_ip1, dt
  # You should make use of the EvaluateDynamics() function to compute f(x,u)
  fi,_ = EvaluateDynamics(planar_arm, context, x_i, u_i)
  fi1, _ = EvaluateDynamics(planar_arm, context, x_ip1, u_ip1)

  s_halfway = (x_i+x_ip1)*0.5 - 0.125*(dt)*(fi1-fi)
  sdot_halfway = 1.5*(x_ip1-x_i)/dt - 0.25*(fi+fi1)
  u_halfway = (u_i+u_ip1)*0.5
  lambda_c_halfway = (lambda_c_i + lambda_c_ip1)*0.5

  dyn, contact = EvaluateDynamics(planar_arm, context, s_halfway, u_halfway)
  h_i = sdot_halfway - dyn
  

  return h_i, contact

def AddCollocationConstraints(prog, planar_arm, context, N, x, u, timesteps):
  n_u = planar_arm.num_actuators()
  n_x = planar_arm.num_positions() + planar_arm.num_velocities()
  n_lambda = 6
  
  for i in range(N - 1):
    def CollocationConstraintHelper(vars):
      x_i = vars[:n_x]
      x_ip1 = vars[n_x:2*n_x]
      u_i = vars[2*n_x: 2*n_x + n_u]
      u_ip1 = vars[2*n_x+n_u:2*(n_x+n_u)]


      return CollocationConstraintEvaluator(planar_arm, context, timesteps[i+1] - timesteps[i], x_i, u_i, x_ip1, u_ip1)
      
    # TODO: Within this loop add the dynamics constraints for segment i (aka collocation constraints)
    #       to prog
    # Hint: use prog.AddConstraint(CollocationConstraintHelper, lb, ub, vars)
    # where vars = hstack(x[i], u[i], ...)
    lower_bound = np.zeros(n_x)
    upper_bound = lower_bound
    prog.AddConstraint(CollocationConstraintHelper, lower_bound, upper_bound, np.hstack([x[i], x[i+1], u[i], u[i+1]]))
