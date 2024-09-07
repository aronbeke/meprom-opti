import numpy as np
from scipy.integrate import odeint
from scipy.optimize import fsolve

'''
Solution-diffusion-electromigration models (SDE)
SDE: Solution-diffusion-electromigration with collocation points (system of algebraic equations)
SDE-DAE: Differential-Algebraic Equation
SDEC: Solution-diffusion-electromigration with coupled flux
SF: solution-friction

spiral wound short-cut: calculations with one bulk concentration
ID: ideal mixing stage

(direct - not shown): concentrations estimated in collocation points
indirect: Polynomial parameters estimated in collocation points

rev: reverse engineering
noflow: only membrane concentrations estimated, cell-side hydrodynamics not included
structural: includes structural properties of membrane module (concentration polarization is more accurate)
'''

def sde_spiral_wound_id(p,args):
    F0,c0,A,l,z,T,k,n,P1,P0,OB2,OB1,OB0,OP1,OP0,beta,dp = args
    '''
    Uses concentration and electric potential as variables across the membrane
    k: number of collocations
    n: number of ions
    molar concentrations

    use:
    mol m3 h Pa K m2

    reduced vectors and matrices do not contain values for x == 0
    we assume phi(x==0) == 0
    '''

    R = 8.314

    Jv = p[0]
    Fr = p[1]
    Fp = p[2]
    j_list = p[3:(3+n)]
    cr_list = p[(3+n):(3+2*n)]
    cm_list = p[(3+2*n):(3+3*n)]
    theta_phi_reduced_list = p[(3+3*n):(3+3*n+(k-1))]
    c_reduced_list = p[(3+3*n+(k-1)):(3+3*n+(k-1)+(k-1)*n)]

    j, cr, cm = np.array([j_list]), np.transpose(np.array([cr_list])), np.transpose(np.array([cm_list]))
    theta_phi_reduced_vector, c_reduced_vector = np.transpose(np.array([theta_phi_reduced_list])), np.array([c_reduced_list])
    l_vector = np.array([l])
    z_vector = np.transpose(np.array([z]))
    c0_vector = np.transpose(np.array([c0]))
    theta_phi_vector = np.vstack((np.array([[0]]),theta_phi_reduced_vector))

    L_matrix = np.tile(l_vector,((k-1),1))
    Z_matrix = np.tile(np.transpose(z_vector),((k-1),1))

    J_shape = ((k-1),n) #Ion flux matrix, each row is the same
    D_shape = (k-1,k) #Polynomial derivatives Vandermonde-matrix at x(j = 2 ... k)
    V_shape = (k,k) #Vandermonde-matrix at x(1 ... k)
    C_shape = (k,n) #Concentration polynomial coefficients
    C_reduced_shape = (k-1,n)
    Phi_shape = (k,n) #Electric potential polynomial coefficients (all columns the same)

    J = np.tile(j,(k-1,1))
    V = np.vander(np.linspace(0,1,k), k, increasing=True)
    V_inv = np.linalg.inv(V)
    D_coeff = np.tile(np.array([range(k)]),(k-1,1))
    D_vander = np.hstack((np.zeros((k-1,1)),np.vander(np.linspace(0,1,k-1), k-1, increasing=True)))
    D = np.multiply(D_coeff,D_vander)
    C_reduced = c_reduced_vector.reshape(C_reduced_shape)
    C = np.vstack((np.transpose(cm),C_reduced))
    Phi = np.tile(theta_phi_vector,(1,n))

    P = P1*(dp) + P0

    omega_br = osmotic_coefficient_brine_tds(cm,OB2,OB1,OB0)
    omega_per = osmotic_coefficient_perm_tds(C[-1,:],OP1,OP0)

    # kxn equations:
    eq_ion_flux = - L_matrix * (D @ (V_inv @ C)) - L_matrix * Z_matrix * C_reduced * (D @ (V_inv @ Phi)) - J
    # k equations:
    eq_neutrality = C_reduced @ z
    # n equations:
    eq_boundary = Jv * np.array([C[-1,:]]) - j
    # 1 eq:
    eq_solvent_flux = P*(dp-R*T*(omega_br*np.sum(cm)-omega_per*np.sum(C[-1,:]))) - Jv
    # n eq:
    # eq_bulk_and_polarization =  np.exp(b0*(Fp/F0)) * cr + (1-np.exp(b0*(Fp/F0))) * (np.sum(np.array([C[-1,:]]).T)/np.sum(cr)) * cr - cm
    eq_bulk_and_polarization =  beta * cr + (1-beta) * (np.sum(np.array([C[-1,:]]).T)/np.sum(cr)) * cr - cm
    # 1 eq:
    eq_permeate_flow = (Jv * A) - Fp
    # 1 eq:
    eq_retentate_flow = F0 - Fp - Fr
    # n eq:
    eq_retentate_concentration = Fr * cr + Fp * np.array([C[-1,:]]).T - F0 * c0_vector

    eq = []
    eq.extend(eq_ion_flux.ravel().tolist())
    eq.extend(eq_neutrality.ravel().tolist())
    eq.extend(eq_boundary.ravel().tolist())
    eq.append(float(eq_solvent_flux))
    eq.extend(eq_bulk_and_polarization.ravel().tolist())
    eq.append(float(eq_permeate_flow))
    eq.append(float(eq_retentate_flow))
    eq.extend(eq_retentate_concentration.ravel().tolist())

    return eq    


def sdec_spiral_wound_id(p,args):
    F0,c0,A,l,z,T,k,n,P1,P0,OB2,OB1,OB0,OP1,OP0,beta,dp,Ki,sf = args
    '''
    Uses concentration and electric potential as variables across the membrane
    k: number of collocations
    n: number of ions
    molar concentrations

    use:
    mol m3 h Pa K m2

    reduced vectors and matrices do not contain values for x == 0
    we assume phi(x==0) == 0
    '''

    R = 8.314

    Jv = p[0]
    Fr = p[1]
    Fp = p[2]
    j_list = p[3:(3+n)]
    cr_list = p[(3+n):(3+2*n)]
    cm_list = p[(3+2*n):(3+3*n)]
    theta_phi_reduced_list = p[(3+3*n):(3+3*n+(k-1))]
    c_reduced_list = p[(3+3*n+(k-1)):(3+3*n+(k-1)+(k-1)*n)]

    j, cr, cm = np.array([j_list]), np.transpose(np.array([cr_list])), np.transpose(np.array([cm_list]))
    theta_phi_reduced_vector, c_reduced_vector = np.transpose(np.array([theta_phi_reduced_list])), np.array([c_reduced_list])
    l_vector = np.array([l])
    z_vector = np.transpose(np.array([z]))
    c0_vector = np.transpose(np.array([c0]))
    theta_phi_vector = np.vstack((np.array([[0]]),theta_phi_reduced_vector))
    Ki_vector = np.array([Ki])

    L_matrix = np.tile(l_vector,((k-1),1))
    Z_matrix = np.tile(np.transpose(z_vector),((k-1),1))
    Ki_matrix = np.tile(Ki_vector,((k-1),1))

    J_shape = ((k-1),n) #Ion flux matrix, each row is the same
    D_shape = (k-1,k) #Polynomial derivatives Vandermonde-matrix at x(j = 2 ... k)
    V_shape = (k,k) #Vandermonde-matrix at x(1 ... k)
    C_shape = (k,n) #Concentration polynomial coefficients
    C_reduced_shape = (k-1,n)
    Phi_shape = (k,n) #Electric potential polynomial coefficients (all columns the same)

    J = np.tile(j,(k-1,1))
    V = np.vander(np.linspace(0,1,k), k, increasing=True)
    V_inv = np.linalg.inv(V)
    D_coeff = np.tile(np.array([range(k)]),(k-1,1))
    D_vander = np.hstack((np.zeros((k-1,1)),np.vander(np.linspace(0,1,k-1), k-1, increasing=True)))
    D = np.multiply(D_coeff,D_vander)
    C_reduced = c_reduced_vector.reshape(C_reduced_shape)
    C = np.vstack((np.transpose(cm),C_reduced))
    Phi = np.tile(theta_phi_vector,(1,n))

    P = P1*(dp) + P0

    omega_br = osmotic_coefficient_brine_tds(cm,OB2,OB1,OB0)
    omega_per = osmotic_coefficient_perm_tds(C[-1,:],OP1,OP0)

    # kxn equations:
    eq_ion_flux = - L_matrix * (D @ (V_inv @ C)) - L_matrix * Z_matrix * C_reduced * (D @ (V_inv @ Phi)) + Ki_matrix * C_reduced * Jv - J
    # k equations:
    eq_neutrality = C_reduced @ z
    # n equations:
    eq_boundary = Jv * np.array([C[-1,:]]) - j
    # 1 eq:
    if sf:
        eq_solvent_flux = P*(dp-R*T*(omega_br*((1-Ki_vector)@cm)-omega_per*((1-Ki_vector)@C[-1,:]))) - Jv
    else:
        eq_solvent_flux = P*(dp-R*T*(omega_br*np.sum(cm)-omega_per*np.sum(C[-1,:]))) - Jv
    # n eq:
    eq_bulk_and_polarization =  beta * cr + (1-beta) * (np.sum(np.array([C[-1,:]]).T)/np.sum(cr)) * cr - cm
    # 1 eq:
    eq_permeate_flow = (Jv * A) - Fp
    # 1 eq:
    eq_retentate_flow = F0 - Fp - Fr
    # n eq:
    eq_retentate_concentration = Fr * cr + Fp * np.array([C[-1,:]]).T - F0 * c0_vector

    eq = []
    eq.extend(eq_ion_flux.ravel().tolist())
    eq.extend(eq_neutrality.ravel().tolist())
    eq.extend(eq_boundary.ravel().tolist())
    eq.append(float(eq_solvent_flux))
    eq.extend(eq_bulk_and_polarization.ravel().tolist())
    eq.append(float(eq_permeate_flow))
    eq.append(float(eq_retentate_flow))
    eq.extend(eq_retentate_concentration.ravel().tolist())

    return eq


def sdec_spiral_wound_id_structural(p,args):
    F0,c0,A,l,z,T,k,kb,n,P1,P0,dp,h,OB2,OB1,OB0,OP1,OP0,rho,eta,l_mesh,df,theta,n_env,b_env,Diff,sigma,Ki = args
    '''
    Uses concentration and electric potential as variables across the membrane
    k: number of collocations
    n: number of ions
    molar concentrations

    use:
    mol m3 h Pa K m2 kg

    reduced vectors and matrices do not contain values for x == 0
    we assume phi(x==0) == 0

    b: boundary
    '''
    omega_p = 0.943
    R = 8.314
    V_sp = 0.5 * np.pi * (df**2) * l_mesh
    V_tot = (l_mesh**2) * h * np.sin(theta)
    epsilon = 1 - (V_sp/V_tot)
    S_vsp = 4 / df
    dh = (4*epsilon) / (2*h + (1-epsilon)*S_vsp)
    v = F0 / (b_env*h*epsilon*n_env)
    Re = (rho*v*dh)/eta

    Diff_vector = np.array([Diff])
    Sc_vector = (1/eta)*(rho*Diff_vector)
    Sh_vector = 0.065 * (Re ** 0.875) * (np.power(Sc_vector,0.25))
    k_mass_vector = (1/dh) * Sh_vector * Diff_vector
    delta_vector = np.divide(Diff_vector,k_mass_vector)

    Jv = p[0]
    Fr = p[1]
    Fp = p[2]
    j_list = p[3:(3+n)]
    cr_list = p[(3+n):(3+2*n)]
    theta_phi_reduced_list = p[(3+2*n):(3+2*n+(k-1))]
    theta_phi_b_reduced_list = p[(3+2*n+(k-1)):(3+2*n+(k-1)+(kb-1))]    
    c_reduced_list = p[(3+2*n+(k-1)+(kb-1)):(3+2*n+(k-1)+(k-1)*n+(kb-1))]
    c_b_reduced_list = p[(3+2*n+(k-1)+(k-1)*n+(kb-1)):(3+2*n+(k-1)+(k-1)*n+(kb-1)+(kb-1)*n)]

    j, cr = np.array([j_list]), np.transpose(np.array([cr_list]))
    theta_phi_reduced_vector, c_reduced_vector = np.transpose(np.array([theta_phi_reduced_list])), np.array([c_reduced_list])
    theta_phi_b_reduced_vector, c_b_reduced_vector = np.transpose(np.array([theta_phi_b_reduced_list])), np.array([c_b_reduced_list])
    l_vector = np.array([l])
    Ki_vector = np.array([Ki])
    z_vector = np.transpose(np.array([z]))
    c0_vector = np.transpose(np.array([c0]))
    theta_phi_vector = np.vstack((np.array([[0]]),theta_phi_reduced_vector))
    theta_phi_b_vector = np.vstack((np.array([[0]]),theta_phi_b_reduced_vector))

    L_matrix = np.tile(l_vector,((k-1),1))
    Ki_matrix = np.tile(Ki_vector,((k-1),1))
    Z_matrix = np.tile(np.transpose(z_vector),((k-1),1))
    Z_b_matrix = np.tile(np.transpose(z_vector),((kb-1),1))
    Diff_matrix = np.tile(Diff_vector,((kb-1),1))
    Delta_matrix = np.tile(delta_vector,((kb-1),1))

    J_shape = ((k-1),n) #Ion flux matrix, each row is the same
    D_shape = (k-1,k) #Polynomial derivatives Vandermonde-matrix at x(j = 2 ... k)
    V_shape = (k,k) #Vandermonde-matrix at x(1 ... k)
    C_shape = (k,n) #Concentration polynomial coefficients
    C_reduced_shape = (k-1,n)
    C_b_reduced_shape = (kb-1,n)
    Phi_shape = (k,n) #Electric potential polynomial coefficients (all columns the same)

    J = np.tile(j,(k-1,1))
    J_b = np.tile(j,(kb-1,1))
    V = np.vander(np.linspace(0,1,k), k, increasing=True)
    V_b = np.vander(np.linspace(0,1,kb), kb, increasing=True)
    V_inv = np.linalg.inv(V)
    V_inv_b = np.linalg.inv(V_b)
    D_coeff = np.tile(np.array([range(k)]),(k-1,1))
    D_coeff_b = np.tile(np.array([range(kb)]),(kb-1,1))
    D_vander = np.hstack((np.zeros((k-1,1)),np.vander(np.linspace(0,1,k-1), k-1, increasing=True)))
    D_vander_b = np.hstack((np.zeros((kb-1,1)),np.vander(np.linspace(0,1,kb-1), kb-1, increasing=True)))
    D = np.multiply(D_coeff,D_vander)
    D_b = np.multiply(D_coeff_b,D_vander_b)
    C_reduced = c_reduced_vector.reshape(C_reduced_shape)
    C_b_reduced = c_b_reduced_vector.reshape(C_b_reduced_shape)

    C_b = np.vstack((cr.T,C_b_reduced))
    C = np.vstack((C_b[-1,:],C_reduced))
    Phi = np.tile(theta_phi_vector,(1,n))
    Phi_b = np.tile(theta_phi_b_vector,(1,n))

    P = P1*(dp) + P0
    cm = C_b[-1,:].flatten().tolist()
    istr = ionic_strength(cm,z)

    omega_m = osmotic_coefficient_brine_tds(cm,OB2,OB1,OB0)
    omega_p = osmotic_coefficient_perm_tds(C[-1,:],OP1,OP0)

    # k-1 x n equations:
    eq_ion_flux = - L_matrix * (D @ (V_inv @ C)) - L_matrix * Z_matrix * C_reduced * (D @ (V_inv @ Phi)) + Ki_matrix * Jv * C_reduced - J
    # kb-1 x n equations:
    eq_boundary_ion_flux = - (1/Delta_matrix) * Diff_matrix * (D_b @ (V_inv_b @ C_b)) - (1/Delta_matrix) * Diff_matrix * Z_b_matrix * C_b_reduced * (D_b @ (V_inv_b @ Phi_b)) + Jv * C_b_reduced - J_b
    # k-1 equations:
    eq_neutrality = C_reduced @ z
    # kb-1 equations:
    eq_boundary_neutrality = C_b_reduced @ z
    # n equations:
    eq_boundary = Jv * np.array([C[-1,:]]) - j
    # 1 eq:
    eq_solvent_flux = P*(dp - sigma * (R*T*(omega_m*np.sum(C_b[-1,:])-omega_p*np.sum(C[-1,:])))) - Jv
    # 1 eq:
    eq_permeate_flow = (Jv * A) - Fp
    # 1 eq:
    eq_retentate_flow = F0 - Fp - Fr
    # n eq:
    eq_retentate_concentration = Fr * cr + Fp * np.array([C[-1,:]]).T - F0 * c0_vector

    eq = []
    eq.extend(eq_ion_flux.ravel().tolist())
    eq.extend(eq_boundary_ion_flux.ravel().tolist())
    eq.extend(eq_neutrality.ravel().tolist())
    eq.extend(eq_boundary_neutrality.ravel().tolist())
    eq.extend(eq_boundary.ravel().tolist())
    eq.append(float(eq_solvent_flux))
    eq.append(float(eq_permeate_flow))
    eq.append(float(eq_retentate_flow))
    eq.extend(eq_retentate_concentration.ravel().tolist())

    return eq

############################################################

def sde_spiral_wound_mesh_module(m,args):

    def out_of_tolerance(beta,beta_new):
        if np.abs(beta_new-beta)/beta <= 0.05:
            return False
        else:
            return True
        
    parameters, constants = args
    F0, c0, A, T, k, n, p0, pp = parameters
    l, z, P1, P0,OB2,OB1,OB0,OP1,OP0, b0, h,rho,l_module,eta,l_mesh,df,theta,n_env,b_env = constants

    pressure_drop = spiral_wound_pressure_drop(h,rho,l_module,eta,F0,l_mesh,df,theta,n_env,b_env)

    pr = p0 - pressure_drop
    dp = pr - pp

    A_bin = A / m

    Cp_matrix = np.zeros((n,m))
    Fp_vector = np.zeros((m,1))

    solutions = []
    ##
    beta = 1
    ##
    beta_has_error = True

    counter = 0
    while beta_has_error and counter < 20:
        # print(beta)
        F0_bin = F0
        c0_bin = c0
        for i in range(m):
            node_data = {}

            r_fact = 1
            m_fact = 1
            zero_fact = 1

            Jv_init = (P1*dp+P0)*dp
            Fp_init = Jv_init*A_bin
            Fr_init = F0_bin - Fp_init
            j_init = (Jv_init*np.array(c0_bin)).tolist()
            cr_init = (r_fact*np.array(c0_bin)).tolist()
            cm_init = (m_fact*np.array(c0_bin)).tolist()
            c_zero_init = (zero_fact*np.array(c0_bin)).tolist()
            init = [Jv_init,Fr_init,Fp_init]
            init.extend(j_init)
            init.extend(cr_init)
            init.extend(cm_init)

            init.extend([0]*(k-1))
            init.extend(c_zero_init*(k-1))

            args = [F0_bin,c0_bin,A_bin,l,z,T,k,n,P1,P0,OB2,OB1,OB0,OP1,OP0,beta,dp]

            node_data['c0'] = c0_bin
            node_data['F0'] = F0_bin
            node_data['P'] = P1*dp + P0

            sol = fsolve(sde_spiral_wound_id, init, args=args)

            node_data['sim'] = sol
            
            Fr_bin = sol[1]
            node_data['Fr'] = Fr_bin
            Fp_bin = sol[2]
            node_data['Fp'] = Fp_bin
            node_data['dp'] = dp
            cm_bin = sol[(3+2*n):(3+3*n)]
            cr_bin = sol[(3+n):(3+2*n)]
            node_data['cm'] = cm_bin
            node_data['cr'] = cr_bin
            c_reduced_bin = sol[(3+3*n+(k-1)):(3+3*n+(k-1)+(k-1)*n)]
            cm = np.transpose(np.array([cm_bin]))
            c_reduced = np.array([c_reduced_bin])
            C_reduced_shape = (k-1,n)
            C_reduced = c_reduced.reshape(C_reduced_shape)
            C = np.vstack((np.transpose(cm),C_reduced))
            cp = np.array([C[-1,:]]).T
            node_data['cp'] = cp.flatten().tolist()

            Cp_matrix[:,i] = cp.flatten()
            Fp_vector[i,0] = Fp_bin
            F0_bin = Fr_bin
            c0_bin = cr_bin

            solutions.append(node_data)
        
        Fr = Fr_bin
        cr_final = cr_bin.flatten().tolist()
        Fp = np.sum(Fp_vector)
        cp = (Cp_matrix @ Fp_vector) / Fp
        cp_final = cp.flatten().tolist()
        beta_new = np.exp(b0*(Fp/F0))
        beta_has_error = out_of_tolerance(beta,beta_new)
        beta = beta_new
        counter += 1

    if counter == 20:
        print('Beta iteration stopped')
    els = {}
    els['Fr'] = Fr
    els['cr'] = cr_final
    els['pr'] = pr
    els['F0'] = F0
    els['c0'] = c0
    els['p0'] = p0
    els['Fp'] = Fp
    els['cp'] = cp_final
    els['pp'] = pp
    els['beta'] = beta
    els['nodes'] = solutions
    els['pressure_drop'] = pressure_drop
    return els


def sdec_spiral_wound_mesh_module(m,args):

    def out_of_tolerance(beta,beta_new):
        if np.abs(beta_new-beta)/beta <= 0.05:
            return False
        else:
            return True
        
    parameters, constants = args
    F0, c0, A, T, k, n, p0, pp = parameters
    l, z, P1, P0,OB2,OB1,OB0,OP1,OP0, b0, h,rho,l_module,eta,l_mesh,df,theta,n_env,b_env,Ki,sf = constants

    pressure_drop = spiral_wound_pressure_drop(h,rho,l_module,eta,F0,l_mesh,df,theta,n_env,b_env)

    pr = p0 - pressure_drop
    dp = pr - pp

    A_bin = A / m

    Cp_matrix = np.zeros((n,m))
    Fp_vector = np.zeros((m,1))

    solutions = []
    ##
    beta = 1
    ##
    beta_has_error = True

    counter = 0
    while beta_has_error and counter < 20:
        # print(beta)
        F0_bin = F0
        c0_bin = c0
        for i in range(m):
            node_data = {}

            r_fact = 1
            m_fact = 1
            zero_fact = 1

            Jv_init = (P1*dp+P0)*dp
            Fp_init = Jv_init*A_bin
            Fr_init = F0_bin - Fp_init
            j_init = (Jv_init*np.array(c0_bin)).tolist()
            cr_init = (r_fact*np.array(c0_bin)).tolist()
            cm_init = (m_fact*np.array(c0_bin)).tolist()
            c_zero_init = (zero_fact*np.array(c0_bin)).tolist()
            init = [Jv_init,Fr_init,Fp_init]
            init.extend(j_init)
            init.extend(cr_init)
            init.extend(cm_init)

            init.extend([0]*(k-1))
            init.extend(c_zero_init*(k-1))

            args = [F0_bin,c0_bin,A_bin,l,z,T,k,n,P1,P0,OB2,OB1,OB0,OP1,OP0,beta,dp,Ki,sf]

            node_data['c0'] = c0_bin
            node_data['F0'] = F0_bin
            node_data['P'] = P1*dp + P0

            sol = fsolve(sdec_spiral_wound_id, init, args=args)

            node_data['sim'] = sol
            
            Fr_bin = sol[1]
            node_data['Fr'] = Fr_bin
            Fp_bin = sol[2]
            node_data['Fp'] = Fp_bin
            node_data['dp'] = dp
            cm_bin = sol[(3+2*n):(3+3*n)]
            cr_bin = sol[(3+n):(3+2*n)]
            node_data['cm'] = cm_bin
            node_data['cr'] = cr_bin
            c_reduced_bin = sol[(3+3*n+(k-1)):(3+3*n+(k-1)+(k-1)*n)]
            cm = np.transpose(np.array([cm_bin]))
            c_reduced = np.array([c_reduced_bin])
            C_reduced_shape = (k-1,n)
            C_reduced = c_reduced.reshape(C_reduced_shape)
            C = np.vstack((np.transpose(cm),C_reduced))
            cp = np.array([C[-1,:]]).T
            node_data['cp'] = cp.flatten().tolist()

            Cp_matrix[:,i] = cp.flatten()
            Fp_vector[i,0] = Fp_bin
            F0_bin = Fr_bin
            c0_bin = cr_bin

            solutions.append(node_data)
        
        Fr = Fr_bin
        cr_final = cr_bin.flatten().tolist()
        Fp = np.sum(Fp_vector)
        cp = (Cp_matrix @ Fp_vector) / Fp
        cp_final = cp.flatten().tolist()
        beta_new = np.exp(b0*(Fp/F0))
        beta_has_error = out_of_tolerance(beta,beta_new)
        beta = beta_new
        counter += 1

    if counter == 20:
        print('Beta iteration stopped')
    els = {}
    els['Fr'] = Fr
    els['cr'] = cr_final
    els['pr'] = pr
    els['F0'] = F0
    els['c0'] = c0
    els['p0'] = p0
    els['Fp'] = Fp
    els['cp'] = cp_final
    els['pp'] = pp
    els['beta'] = beta
    els['nodes'] = solutions
    els['pressure_drop'] = pressure_drop
    return els


def sde_spiral_wound_structural_mesh_module(m,args):
    F0,c0,A,l,z,T,k,kb,n,P1,P0,dp,h,rho,l_module,eta,l_mesh,df,theta,n_env,b_env,Diff,sigma,Ki = args

    A_bin = A / m
    l_bin = l_module / m

    F0_bin = F0
    c0_bin = c0
    dp_bin = dp

    Cp_matrix = np.zeros((n,m))
    Fp_vector = np.zeros((m,1))

    for i in range(m):
        pressure_drop_bin = spiral_wound_pressure_drop(h,rho,l_bin,eta,F0_bin,l_mesh,df,theta,n_env,b_env)
        dp_bin = dp_bin - pressure_drop_bin

        Jv_init = (P1*dp_bin+P0)*dp_bin
        Fp_init = Jv_init*A_bin
        Fr_init = F0_bin - Fp_init
        j_init = (Jv_init*np.array(c0_bin)).tolist()
        cr_init = (1.1*np.array(c0_bin)).tolist()
        c_zero_init = (0.5*np.array(c0_bin)).tolist()
        init = [Jv_init,Fr_init,Fp_init]
        init.extend(j_init)
        init.extend(cr_init)

        init.extend([0]*(k-1))
        init.extend([0]*(kb-1))
        init.extend(c_zero_init*(k-1))
        init.extend(cr_init*(kb-1))

        args = [F0_bin,c0_bin,A_bin,l,z,T,k,kb,n,P1,P0,dp_bin,h,rho,eta,l_mesh,df,theta,n_env,b_env,Diff,sigma,Ki]

        sol = fsolve(sdec_spiral_wound_id_structural, init, args=args)
        
        Fr_bin = sol[1]
        Fp_bin = sol[2]
        cr_bin = sol[(3+n):(3+2*n)]
        c_reduced_bin = sol[(3+2*n+(k-1)+(kb-1)):(3+2*n+(k-1)+(k-1)*n+(kb-1))]
        c_b_reduced_bin = sol[(3+2*n+(k-1)+(k-1)*n+(kb-1)):(3+2*n+(k-1)+(k-1)*n+(kb-1)+(kb-1)*n)]
        c_reduced = np.array([c_reduced_bin])
        c_b_reduced = np.array([c_b_reduced_bin])
        C_reduced_shape = (k-1,n)
        C_b_reduced_shape = (kb-1,n)
        C_reduced = c_reduced.reshape(C_reduced_shape)
        C_b_reduced = c_b_reduced.reshape(C_b_reduced_shape)
        cp = np.array([C_reduced[-1,:]]).T

        polarization = np.divide(C_b_reduced[-1,:]-cp.T,np.array(cr_bin)-cp.T)
        #print(polarization)

        Cp_matrix[:,i] = cp.flatten()
        Fp_vector[i,0] = Fp_bin
        F0_bin = Fr_bin
        c0_bin = cr_bin
    
    Fr = Fr_bin
    cr_final = cr_bin.flatten().tolist()
    Fp = np.sum(Fp_vector)
    cp = (Cp_matrix @ Fp_vector) / Fp
    cp_final = cp.flatten().tolist()
    pressure_drop = dp - dp_bin

    #print(Fp_vector)
    return Fr, Fp, cr_final, cp_final, pressure_drop

#####################

def spiral_wound_pressure_drop(h,rho,l,eta,Qf,l_m,df,theta,n,b):
    '''
    rho         density             kg/m3
    l           envelope length     m
    eta         dynamic viscosity   kg/(mh)
    Qf          feed flow rate      m3/h
    l_m         mesh size           m
    df          filament thickness  m
    theta       angle               rad
    n           no. of envelopes    -
    b           width of envelope   m

    returns Pascal
    '''

    V_sp = 0.5 * np.pi * (df**2) * l_m
    V_tot = (l_m**2) * h * np.sin(theta)
    epsilon = 1 - (V_sp/V_tot)
    S_vsp = 4 / df
    dh = (4*epsilon) / (2*h + (1-epsilon)*S_vsp)
    v = Qf / (b*h*epsilon*n)
    Re = (rho*v*dh)/eta

    pd = (6.23*(Re**(-0.3))*rho*(v**2)*l) / (2*dh*(3600**2))

    return pd


def osmotic_coefficient_brine_i(I):
    return 0.0214*(I**2) - 0.0503 * I + 0.9567

def osmotic_coefficient_brine_tds(c,OB2,OB1,OB0):
    sumc = np.sum(c)
    return OB2*(sumc**2) + OB1 * sumc + OB0

def osmotic_coefficient_perm_tds(c,OP1,OP0):
    sumc = np.sum(c)
    return OP1*(sumc) + OP0


def ionic_strength(c,z):
    '''
    c: mol/m3
    '''
    z_arr = np.array(z)
    z2_arr = np.power(z_arr,2)
    I = 0.5 * np.dot((1/1000)*np.array(c),z2_arr)
    return I