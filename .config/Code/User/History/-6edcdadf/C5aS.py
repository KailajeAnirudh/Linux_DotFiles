import numpy as np
from pydrake.multibody.all import JacobianWrtVariable
import pydrake.math
from pydrake.autodiffutils import AutoDiffXd

def EvaluateDynamics(AD_plant_local, AD_plant_context, state, control, contact_force):
    AD_plant_local.SetPositionsAndVelocities(AD_plant_context, state)

    M = AD_plant_local.CalcMassMatrixViaInverseDynamics(AD_plant_context)
    B = AD_plant_local.MakeActuationMatrix()
    g = AD_plant_local.CalcGravityGeneralizedForces(AD_plant_context)
    C = AD_plant_local.CalcBiasTerm(AD_plant_context)

    Jleft = AD_plant_local.CalcJacobianTranslationalVelocity(AD_plant_context, JacobianWrtVariable.kV, 
                                                            AD_plant_local.GetFrameByName("left_lower_leg"), np.array([0,0, -0.5]), 
                                                            AD_plant_local.world_frame())
    Jright = AD_plant_local.CalcJacobianTranslationalVelocity(AD_plant_context, JacobianWrtVariable.kV, 
                                                            AD_plant_local.GetFrameByName("right_lower_leg"), np.array([0,0, -0.5]), 
                                                            AD_plant_local.world_frame())
    JdotVleft = AD_plant_local.CalcBiasTranslationalAcceleration(AD_plant_context, JacobianWrtVariable.kV,
                                                            AD_plant_local.GetFrameByName("left_lower_leg"), np.array([0,0,-0.5]),
                                                            AD_plant_local.world_frame())
    JdotVright = AD_plant_local.CalcBiasTranslationalAcceleration(AD_plant_context, JacobianWrtVariable.kV,
                                                            AD_plant_local.GetFrameByName("right_lower_leg"), np.array([0,0,-0.5]),
                                                            AD_plant_local.world_frame())
    
    J_c = np.row_stack((Jleft, Jright)); J_c_dot_v = np.row_stack((JdotVleft, JdotVright))

    if (state.dtype == AutoDiffXd):
        M_inv = pydrake.math.inv(state)
    else:
        M_inv = np.linalg.inv(M)
    v_dot = M_inv @(B@control+g-C+J_c.T@contact_force)
    x_dot = np.hstack((state[-AD_plant_local.num_velocities():], v_dot))

    return x_dot

def EvaluateFootAcc(AD_plant_local, AD_plant_context, state, control, contact_force):
    AD_plant_local.SetPositionsAndVelocities(AD_plant_context, state)

    M = AD_plant_local.CalcMassMatrixViaInverseDynamics(AD_plant_context)
    B = AD_plant_local.MakeActuationMatrix()
    g = AD_plant_local.CalcGravityGeneralizedForces(AD_plant_context)
    C = AD_plant_local.CalcBiasTerm(AD_plant_context)

    Jleft = AD_plant_local.CalcJacobianTranslationalVelocity(AD_plant_context, JacobianWrtVariable.kV, 
                                                            AD_plant_local.GetFrameByName("left_lower_leg"), np.array([0,0, -0.5]), 
                                                            AD_plant_local.world_frame())
    Jright = AD_plant_local.CalcJacobianTranslationalVelocity(AD_plant_context, JacobianWrtVariable.kV, 
                                                            AD_plant_local.GetFrameByName("right_lower_leg"), np.array([0,0, -0.5]), 
                                                            AD_plant_local.world_frame())
    JdotVleft = AD_plant_local.CalcBiasTranslationalAcceleration(AD_plant_context, JacobianWrtVariable.kV,
                                                            AD_plant_local.GetFrameByName("left_lower_leg"), np.array([0,0,-0.5]),
                                                            AD_plant_local.world_frame())
    JdotVright = AD_plant_local.CalcBiasTranslationalAcceleration(AD_plant_context, JacobianWrtVariable.kV,
                                                            AD_plant_local.GetFrameByName("right_lower_leg"), np.array([0,0,-0.5]),
                                                            AD_plant_local.world_frame())
    
    J_c = np.row_stack((Jleft, Jright)); J_c_dot_v = np.row_stack((JdotVleft, JdotVright))

    if (state.dtype == AutoDiffXd):
        M_inv = pydrake.math.inv(state)
    else:
        M_inv = np.linalg.inv(M)
    v_dot = M_inv @(B@control+g-C+J_c.T@contact_force)
    x_dot = np.hstack((state[-AD_plant_local.num_velocities():], v_dot))
    foot_acceleration = (J_c @ v_dot).reshape(-1, 1) + J_c_dot_v
    
    return foot_acceleration

def EvaluateFootPos(AD_plant_local, AD_plant_context, state):
    AD_plant_local.SetPositionsAndVelocities(AD_plant_context, state)
    left_foot_zpos = AD_plant_local.CalcPointsPositions(AD_plant_context, 
                                                        AD_plant_local.GetFrameByName("left_lower_leg"), np.array([0,0,-0.5]), 
                                                        AD_plant_local.world_frame())[2]
    right_foot_zpos = AD_plant_local.CalcPointsPositions(AD_plant_context, 
                                                         AD_plant_local.GetFrameByName("right_lower_leg"), np.array([0,0,-0.5]), 
                                                         AD_plant_local.world_frame())[2]
    foot_zpos = np.row_stack((left_foot_zpos, right_foot_zpos))

    return foot_zpos


def CollocationConstraintEvaluator(AD_plant_local, AD_plant_context, state_i, state_i1, control_i, control_i1, lambda_i, lambda_i1, lambda_col, dt):
        
    fi, = EvaluateDynamics(AD_plant_local, AD_plant_context, state_i, control_i, lambda_i)
    fi1 = EvaluateDynamics(AD_plant_local, AD_plant_context, state_i1, control_i1, lambda_i1)

    footpos_i = EvaluateFootPos(AD_plant_local, AD_plant_context, state_i)
    footacc_i = EvaluateFootAcc(AD_plant_local, AD_plant_context, state_i, control_i, lambda_i)
    


    collocation_state = 0.5*(state_i+state_i1) - 0.125*dt*(fi1-fi)
    collocation_control = 0.5*(control_i+control_i1)
    collocation_state_dot = 1.5*(state_i1-state_i)/dt -0.25*(fi+fi1)

    fcol, footacc_col, footpos_col = EvaluateDynamics(AD_plant_local, AD_plant_context, collocation_state, collocation_control, lambda_col)
    footacc_col = EvaluateFootAcc(AD_plant_local, AD_plant_context, collocation_state, collocation_control, lambda_col)
    footpos_col = EvaluateFootPos(AD_plant_local, AD_plant_context, collocation_state, collocation_control, lambda_col)

    h_i = collocation_state_dot-fcol
    colloction_constraints_params = np.hstack(([h_i, footacc_i, footacc_col, footpos_i, footpos_col]))

    return colloction_constraints_params

def LastPointConstraintEvaluator(AD_plant_local, AD_plant_context, state_i, control_i, lambda_i, dt):
    footacc = EvaluateFootAcc(AD_plant_local, AD_plant_context, state_i, control_i, lambda_i)
    footpos = EvaluateFootPos(AD_plant_local, AD_plant_context, state_i, control_i, lambda_i)

    return np.hstack(([footacc, footpos]))