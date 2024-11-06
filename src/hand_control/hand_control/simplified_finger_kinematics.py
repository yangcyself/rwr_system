from cmath import pi
from re import T
import numpy as np
from scipy.optimize import fsolve
import sympy as sym
from sympy.solvers import solve
import matplotlib.pyplot as plt

import sympy as sp
from sympy.utilities.lambdify import lambdify

################################################################################
#TODO: Make it specific to your hand
################################################################################

def tendon_length_rolling_contact(theta, alpha1_deg, alpha2_deg, R1, R2):
   """Computes tendon length for a given theta
   """
   alpha1 = np.deg2rad(alpha1_deg)
   alpha2 = np.deg2rad(alpha2_deg)
   
   theta_scaled = theta * R2/(R1+R2)

   # Calculate centers
   center1 = sp.Matrix([0, 0])
   center2_initial = sp.Matrix([R1 + R2, 0])
   rotation_matrix = sp.Matrix([[sp.cos(theta_scaled), -sp.sin(theta_scaled)], 
                              [sp.sin(theta_scaled), sp.cos(theta_scaled)]])
   center2 = rotation_matrix * center2_initial
   
   # Calculate attachment points
   attach1 = sp.Matrix([R1 * sp.cos(alpha1), R1 * sp.sin(alpha1)])
   attach2_initial = sp.Matrix([center2_initial[0] + R2 * sp.cos(sp.pi - alpha2),
                              center2_initial[1] + R2 * sp.sin(sp.pi - alpha2)])
   attach2_prime = rotation_matrix * attach2_initial
   theta2_scaled = theta_scaled * (R1 / R2)
   rotation_matrix2 = sp.Matrix([[sp.cos(theta2_scaled), -sp.sin(theta2_scaled)], 
                                 [sp.sin(theta2_scaled), sp.cos(theta2_scaled)]])
   attach2 = center2 + rotation_matrix2 * (attach2_prime - center2)
   
   # Calculate tendon length
   tendon_length = sp.sqrt((attach2[0] - attach1[0])**2 + (attach2[1] - attach1[1])**2)

   return tendon_length


def tendon_length_pin_joint(theta, R1):
   """Computes tendon length for a given theta
   """
   return -R1 * theta


def get_tendon_lengths_lambda(theta1, theta2, theta3, muscle_group):
   tendon_lengths = []
   alpha_deg = muscle_group.alpha
   radius_mm = muscle_group.joint_radius

   if muscle_group.name == "thumb":

      # CMC abduction (thuumb)
      tl0 = tendon_length_pin_joint(theta1, radius_mm[0][0])
      tl0 = sp.simplify(tl0)

      # CMC flexion (thumb)
      tl1 = tendon_length_pin_joint(theta2, radius_mm[1][0])
      tl1 = sp.simplify(tl1)

      # PIP flexion
      tl2 = tendon_length_rolling_contact(theta3, alpha_deg[2][0], alpha_deg[2][1], radius_mm[2][0], radius_mm[2][1])
      tl2 = sp.simplify(tl2)

      # Each joint is pretty much independent
      tendon_lengths = [tl0, tl1, tl2]
   else:
      # MCP abduction
      tl0 = tendon_length_rolling_contact(-theta1, alpha_deg[0][0], alpha_deg[0][1], radius_mm[0][0], radius_mm[0][1])
      tl0 = sp.simplify(tl0)

      # MCP flexion
      tl1 = tendon_length_rolling_contact(-theta2, alpha_deg[1][0], alpha_deg[1][1], radius_mm[1][0], radius_mm[1][1])
      tl1 = sp.simplify(tl1)

      # PIP flexion
      tl2 = tendon_length_rolling_contact(theta3, alpha_deg[2][0], alpha_deg[2][1], radius_mm[2][0], radius_mm[2][1])
      tl2 = sp.simplify(tl2)

      # Add MCP abduction length to MCP flexion length
      # since they share tendons for the differential actuation
      tendon_lengths = [tl0 - tl1, - tl0 - tl1, tl2]

   return tendon_lengths
