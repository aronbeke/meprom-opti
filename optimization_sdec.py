import numpy as np
import pyomo
from pyomo.environ import *
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
from pyomo.util.infeasible import log_infeasible_constraints

#########

def initialize_model(pm):
    #ns pm['solutes']
    #nk pm['coll']ocation
    #ne elements in a stage
    #nst pm['stages'] in total
    #nn pm['nodes'] in an element
    pm['A_node'] = pm['A'] / pm['nn']

    # FYI
    
    '''
    J_shape = ((pm['nk']-1),pm['ns']) #Ion flux matrix, each row is the same
    D_shape = (pm['nk']-1,pm['nk']) #Polynomial derivatives Vandermonde-matrix at x(j = 2 ... k)
    V_shape = (pm['nk'],pm['nk']) #Vandermonde-matrix at x(1 ... k)
    C_shape = (pm['nk'],pm['ns']) #Concentration polynomial coefficients
    C_reduced_shape = (pm['nk']-1,pm['ns'])
    Phi_shape = (pm['nk'],pm['ns']) #Electric potential polynomial coefficients (all columns the same)
    '''

    V = np.vander(np.linspace(0,1,pm['nk']), pm['nk'], increasing=True)
    V_inv = np.linalg.inv(V)
    D_coeff = np.tile(np.array([range(pm['nk'])]),(pm['nk']-1,1))
    D_vander = np.hstack((np.zeros((pm['nk']-1,1)),np.vander(np.linspace(0,1,pm['nk']-1), pm['nk']-1, increasing=True)))
    D = np.multiply(D_coeff,D_vander)

    pm['V_sp'] = 0.5 * np.pi * (pm['df']**2) * pm['l_mesh']
    pm['V_tot'] = (pm['l_mesh']**2) * pm['h'] * np.sin(pm['theta'])
    pm['epsilon'] = 1 - (pm['V_sp']/pm['V_tot'])
    pm['S_vsp'] = 4 / pm['df']
    pm['dh'] = (4*pm['epsilon']) / (2*pm['h'] + (1-pm['epsilon'])*pm['S_vsp'])
    pm['v_factor'] = 1 / (pm['b_env']*pm['h']*pm['epsilon']*pm['n_env'])
    pm['Re_factor'] = (pm['rho']*pm['dh'])/pm['eta']

    pm['L'] = np.array(pm['Li_list'])
    pm['K'] = np.array(pm['Ki_list'])
    pm['z'] = np.array(pm['zi_list'])

    pm['DVi_matrix'] = D @ V_inv

    # Sets
    pm['stages'] = range(pm['nst'])
    pm['elems'] = range(pm['ne']) # elements
    pm['red_elems'] = range(pm['ne']-1)
    pm['nodes'] = range(pm['nn'])
    pm['red_nodes'] = range(pm['nn']-1)
    pm['solutes'] = range(pm['ns'])
    pm['coll'] = range(pm['nk'])
    pm['red_coll'] = range(1,pm['nk'])
    # pm['stages'] = RangeSet(num_pm['stages'])

    # Initializations
    C_init = {}
    for i in pm['coll']:
        for j in pm['solutes']:
            C_init[(i,j)] = pm['c_feed_dict'][j]

    dCdx_init = {}
    for i in pm['red_coll']:
        for j in pm['solutes']:
            dCdx_init[(i,j)] = pm['c_feed_dict'][j]

    Phi_init = {}
    for i in pm['coll']:
        Phi_init[i] = 0

    dPhidx_init = {}
    for i in pm['red_coll']:
        Phi_init[i] = 0

    J_init = {}
    for i in pm['solutes']:
        J_init[i] = 0


    # Bounds
    C_bound = {}
    # C_bound[0] = (0,80)
    # C_bound[1] = (0,500)
    # C_bound[2] = (0,1500)
    # C_bound[3] = (0,40)
    # C_bound[4] = (0,400)
    # C_bound[5] = (0,1200)
    # C_bound[6] = (0,15)

    C_bound[0] = (0,200)
    C_bound[1] = (0,1000)
    C_bound[2] = (0,2000)
    C_bound[3] = (0,100)
    C_bound[4] = (0,500)
    C_bound[5] = (0,2000)
    C_bound[6] = (0,100)

    C_bound_mx = {}
    for i in pm['coll']:
        for j in pm['solutes']:
            C_bound_mx[(i,j)] = C_bound[j]

    dCdx_bound_mx = {}
    for i in pm['red_coll']:
        for j in pm['solutes']:
            dCdx_bound_mx[(i,j)] = (-C_bound[j][1]*10,C_bound[j][1]*10)

    J_bound = {}
    for i in pm['solutes']:
        J_bound[i] = (0,(pm['F_lim']*C_bound[i][1])/(pm['A_node']))

    pm['bounds'] = {'C': C_bound, 'C_mx' : C_bound_mx, 'dCdx': dCdx_bound_mx, 'J': J_bound}
    pm['inits'] = {'C': C_init, 'dCdx': dCdx_init, 'Phi': Phi_init, 'dPhidx': dPhidx_init,'J': J_init}



# BLOCK RULES ################################################################
    
def sedc_node_rule(b,pm):     
    b.P = Var(within=NonNegativeReals, bounds = (0,0.008), initialize = 0.004)
    b.omega_brine = Var(within=NonNegativeReals, bounds = (0,1.5), initialize = 0.95)
    b.omega_perm = Var(within=NonNegativeReals, bounds = (0,1.5), initialize = 0.95)
    b.beta = Var(within=NonNegativeReals, bounds = (0,3), initialize = 1.0)
    b.dp = Var(within=NonNegativeReals, bounds = (0,pm['dp_max']), initialize = 0)
    b.Fr = Var(within=NonNegativeReals, bounds = (0,pm['F_lim']), initialize = pm['F_feed'])
    b.Fp = Var(within=NonNegativeReals, bounds = (0,pm['F_lim']), initialize = 0)
    b.J = Var(pm['solutes'], within=NonNegativeReals, bounds = lambda b, i: pm['bounds']['J'][i], initialize = pm['inits']['J'])
    b.Jv = Var(within=NonNegativeReals, bounds = (0,pm['F_lim']/(pm['A_node'])), initialize = 0.0)
    b.C = Var(pm['coll'], pm['solutes'], within=NonNegativeReals, bounds = lambda b, i, j: pm['bounds']['C_mx'][(i,j)], initialize = pm['inits']['C'])
    b.Phi = Var(pm['coll'], within=Reals, bounds = (-1,1),initialize = pm['inits']['Phi'])
    b.F0 = Var(within=NonNegativeReals, bounds = (0,pm['F_lim']), initialize = pm['F_feed'])
    b.C0 = Var(pm['solutes'], within=NonNegativeReals, bounds = lambda b, i: pm['bounds']['C'][i], initialize = pm['c_feed_dict'])
    b.Cr = Var(pm['solutes'], within=NonNegativeReals, bounds = lambda b, i: pm['bounds']['C'][i], initialize = pm['c_feed_dict'])
    b.dPhidx = Var(pm['red_coll'], within=Reals, bounds = (-10,10), initialize = pm['inits']['dPhidx'])
    b.dCdx = Var(pm['red_coll'],pm['solutes'], within=Reals, bounds = lambda b, i, j: pm['bounds']['dCdx'][(i,j)],initialize = pm['inits']['dCdx'])
    
    def electroneutrality_rule(b,c):
        return sum(b.C[c,i]*pm['z'][i] for i in pm['solutes']) == 0
    
    def permeate_boundary_rule(b,s):
        return b.J[s] == b.Jv * b.C[pm['nk']-1,s]

    def ion_flux_rule(b, c, s):
        return b.J[s] == -pm['L'][s] * b.dCdx[c,s] - pm['L'][s] * pm['z'][s] * b.C[c, s] * b.dPhidx[c] + pm['K'][s] * b.C[c, s] * b.Jv

    def solvent_flux_rule(b):
        return b.Jv == b.P * (b.dp - 1e-5*8.314*pm['T']*(b.omega_brine*sum(b.C[0,i] for i in pm['solutes']) - b.omega_perm*sum(b.C[pm['nk']-1,j] for j in pm['solutes'])))
    
    def concentration_polarization_rule(b,s):
        return b.C[0,s] == b.Cr[s] * (b.beta + (1-b.beta)*(sum(b.C[pm['nk']-1,j] for j in pm['solutes'])/sum(b.Cr[i] for i in pm['solutes'])))

    def solute_mass_balance_rule(b,s):
        return b.F0 * b.C0[s] == b.Fr * b.Cr[s] + b.Fp * b.C[pm['nk']-1,s]
    
    def overall_mass_balance_rule(b):
        return b.F0 == b.Fr + b.Fp
    
    def permeate_stream_rule(b):
        return b.Fp == b.Jv * pm['A_node']
    
    def potential_rule(b):
        return b.Phi[0] == 0.0
    
    def phi_derivative_rule(b,c):
        return b.dPhidx[c] == sum(pm['DVi_matrix'][c-1][i] * b.Phi[i] for i in pm['coll'])

    def concentration_derivative_rule(b,c,s):
        return b.dCdx[c,s] == sum(pm['DVi_matrix'][c-1][i] * b.C[i,s] for i in pm['coll'])
    
    def permeance_rule(b):
        return b.P == pm['P1'] * b.dp + pm['P0']
    
    def omega_brine_rule(b):
        return b.omega_brine == pm['OB2'] * sum(b.C[0,i] for i in pm['solutes'])**2 + pm['OB1'] * sum(b.C[0,i] for i in pm['solutes']) + pm['OB0']
    
    def omega_perm_rule(b):
        return b.omega_perm == pm['OP1'] * sum(b.C[pm['nk']-1,j] for j in pm['solutes']) + pm['OP0']
    
    # def numerical_stability(b):
    #     return sum(b.C[0,i] for i in pm['solutes']) - sum(b.C[pm['nk']-1,j] for j in pm['solutes']) >= 0

    
    b.electroneutrality = Constraint(pm['red_coll'],rule = electroneutrality_rule)
    b.permeate_boundary = Constraint(pm['solutes'], rule = permeate_boundary_rule)
    b.ion_flux = Constraint(pm['red_coll'],pm['solutes'], rule = ion_flux_rule)
    b.solvent_flux = Constraint(rule = solvent_flux_rule)
    b.solute_mass_balance = Constraint(pm['solutes'], rule = solute_mass_balance_rule)
    b.concentration_polarization = Constraint(pm['solutes'], rule = concentration_polarization_rule)
    b.overall_mass_balance = Constraint(rule = overall_mass_balance_rule)
    b.permeate_stream = Constraint(rule = permeate_stream_rule)
    b.phi_derivative = Constraint(pm['red_coll'],rule = phi_derivative_rule)
    b.c_derivative = Constraint(pm['red_coll'],pm['solutes'],rule = concentration_derivative_rule)
    b.potential_const = Constraint(rule = potential_rule)
    b.permeance_const = Constraint(rule = permeance_rule)
    b.omega_brine_const = Constraint(rule = omega_brine_rule)
    b.omega_perm_const = Constraint(rule = omega_perm_rule)
    # b.stability_const = Constraint(rule = numerical_stability)

def element_rule(b,pm):

    b.p0 = Var(within=NonNegativeReals, bounds = (0,pm['dp_max']), initialize = pm['dp_max'])
    b.pp = Var(within=NonNegativeReals, bounds = (0,pm['dp_max']), initialize = pm['dp_max'])
    b.pdrop = Var(within=NonNegativeReals, bounds = (0,pm['dp_max']), initialize = 0)
    # b.dp = Var(within=NonNegativeReals, bounds = (0,pm['dp_max']), initialize = 0.0)
    b.beta = Var(within=NonNegativeReals, bounds = (0,3), initialize = 1.0)
    b.nodes = Block(pm['nodes'],rule=lambda b: sedc_node_rule(b,pm))
    b.F0 = Var(within=NonNegativeReals, bounds = (0,pm['F_lim']), initialize = pm['F_feed'])
    b.C0 = Var(pm['solutes'], within=NonNegativeReals, bounds = lambda b, i: pm['bounds']['C'][i], initialize = pm['c_feed_dict'])
    b.Fr = Var(within=NonNegativeReals, bounds = (0,pm['F_lim']), initialize = pm['F_feed'])
    b.Cr = Var(pm['solutes'], within=NonNegativeReals, bounds = lambda b, i: pm['bounds']['C'][i], initialize = pm['c_feed_dict'])
    b.Fp = Var(within=NonNegativeReals, bounds = (0,pm['F_lim']), initialize = 0)
    b.Cp = Var(pm['solutes'], within=NonNegativeReals, bounds = lambda b, i: pm['bounds']['C'][i], initialize = pm['c_feed_dict'])

    def intermediary_retentate_rule(b,n):
        return b.nodes[n].Fr == b.nodes[n+1].F0

    def intermediary_retentate_concentration_rule(b,n,s):
        return b.nodes[n].Cr[s] == b.nodes[n+1].C0[s]
    
    def feed_rule(b):
        return b.F0 == b.nodes[0].F0
    
    def feed_concentration_rule(b,s):
        return b.C0[s] == b.nodes[0].C0[s]

    def retentate_rule(b):
        return b.Fr == b.nodes[pm['nn']-1].Fr

    def retentate_concentration_rule(b,s):
        return b.Cr[s] == b.nodes[pm['nn']-1].Cr[s]
    
    def permeate_rule(b):
        return b.Fp == sum(b.nodes[i].Fp for i in pm['nodes'])
    
    def permeate_concentration_rule(b,s):
        return b.Fp * b.Cp[s] == sum(b.nodes[i].Fp * b.nodes[i].C[pm['nk']-1,s] for i in pm['nodes'])
    
    # def node_pressure_rule(b,n):
    #     return b.dp == b.nodes[n].dp
    
    def node_pressure_rule(b,n):
        return b.nodes[n].dp == b.p0 - b.pp - b.pdrop
    
    def pressure_drop_rule(b):
        return b.pdrop == (1/100000)*(6.23*((pm['Re_factor']*pm['v_factor']*b.F0)**(-0.3))*pm['rho']*((pm['v_factor']*b.F0)**2)*pm['l_module']) / (2*pm['dh']*(3600**2))
    
    # def pressure_drop_rule(b):
    #     return b.pdrop == 0
    
    def polarization_rule(b):
        return b.beta == exp(pm['b0']*(b.Fp/b.F0))
    
    def beta_node_rule(b,n):
        return b.beta == b.nodes[n].beta
    
    b.intermediary_retentate = Constraint(pm['red_nodes'], rule = intermediary_retentate_rule)
    b.intermediary_retentate_concentration = Constraint(pm['red_nodes'], pm['solutes'], rule = intermediary_retentate_concentration_rule)
    b.feed = Constraint(rule = feed_rule)
    b.feed_concentration = Constraint(pm['solutes'], rule = feed_concentration_rule)
    b.retentate = Constraint(rule = retentate_rule)
    b.retentate_concentration = Constraint(pm['solutes'], rule = retentate_concentration_rule)
    b.permeate = Constraint(rule = permeate_rule)
    b.permeate_concentration = Constraint(pm['solutes'], rule = permeate_concentration_rule)

    b.node_pressure = Constraint(pm['nodes'], rule = node_pressure_rule)
    b.pressure_drop = Constraint(rule = pressure_drop_rule)
    
    b.polarization = Constraint(rule = polarization_rule)
    b.beta_node = Constraint(pm['nodes'],rule = beta_node_rule)


def stage_rule(b,pm):

    b.p0 = Var(within=NonNegativeReals, bounds = (0,pm['dp_max']), initialize = pm['dp_max'])
    b.pp = Var(within=NonNegativeReals, bounds = (0,pm['dp_max']), initialize = pm['dp_max'])
    b.pr = Var(within=NonNegativeReals, bounds = (0,pm['dp_max']), initialize = pm['dp_max'])
    # b.dp = Var(within=NonNegativeReals, bounds = (0,pm['dp_max']), initialize = 0.0)
    b.elems = Block(pm['elems'],rule=lambda b: element_rule(b,pm))
    b.F0 = Var(within=NonNegativeReals, bounds = (0,pm['F_lim']), initialize = pm['F_feed'])
    b.C0 = Var(pm['solutes'], within=NonNegativeReals, bounds = lambda b, i: pm['bounds']['C'][i], initialize = pm['c_feed_dict'])
    b.Fr = Var(within=NonNegativeReals, bounds = (0,pm['F_lim']), initialize = pm['F_feed'])
    b.Cr = Var(pm['solutes'], within=NonNegativeReals, bounds = lambda b, i: pm['bounds']['C'][i], initialize = pm['c_feed_dict'])
    b.Fp = Var(within=NonNegativeReals, bounds = (0,pm['F_lim']), initialize = 0)
    b.Cp = Var(pm['solutes'], within=NonNegativeReals, bounds = lambda b, i: pm['bounds']['C'][i], initialize = pm['c_feed_dict'])

    def intermediary_retentate_rule(b,n):
        return b.elems[n].Fr == b.elems[n+1].F0

    def intermediary_retentate_concentration_rule(b,n,s):
        return b.elems[n].Cr[s] == b.elems[n+1].C0[s]
    
    def feed_rule(b):
        return b.F0 == b.elems[0].F0
    
    def feed_concentration_rule(b,s):
        return b.C0[s] == b.elems[0].C0[s]

    def retentate_rule(b):
        return b.Fr == b.elems[pm['ne']-1].Fr

    def retentate_concentration_rule(b,s):
        return b.Cr[s] == b.elems[pm['ne']-1].Cr[s]
    
    def permeate_rule(b):
        return b.Fp == sum(b.elems[i].Fp for i in pm['elems'])
    
    def permeate_concentration_rule(b,s):
        return b.Fp * b.Cp[s] == sum(b.elems[i].Fp * b.elems[i].Cp[s] for i in pm['elems'])
    
    # def elem_pressure_rule(b,n):
    #     return b.dp == b.elems[n].dp
    
    def intermediary_pressure_rule(b,n):
        return b.elems[n+1].p0 == b.elems[n].p0 - b.elems[n].pdrop

    def feed_pressure_rule(b):
        return b.p0 == b.elems[0].p0

    def retentate_pressure_rule(b):
        return b.pr == b.elems[pm['ne']-1].p0 - b.elems[pm['ne']-1].pdrop
    
    def permeate_pressure_rule(b,n):
        return b.pp == b.elems[n].pp
    
    b.intermediary_retentate = Constraint(pm['red_elems'], rule = intermediary_retentate_rule)
    b.intermediary_retentate_concentration = Constraint(pm['red_elems'], pm['solutes'], rule = intermediary_retentate_concentration_rule)
    b.feed = Constraint(rule = feed_rule)
    b.feed_concentration = Constraint(pm['solutes'], rule = feed_concentration_rule)
    b.retentate = Constraint(rule = retentate_rule)
    b.retentate_concentration = Constraint(pm['solutes'], rule = retentate_concentration_rule)
    b.permeate = Constraint(rule = permeate_rule)
    b.permeate_concentration = Constraint(pm['solutes'], rule = permeate_concentration_rule)

    b.feed_pressure = Constraint(rule = feed_pressure_rule)
    b.intermediary_pressure = Constraint(pm['red_elems'], rule = intermediary_pressure_rule)
    b.retentate_pressure = Constraint(rule = retentate_pressure_rule)
    b.permeate_pressure = Constraint(pm['elems'],rule = permeate_pressure_rule)
    # b.elem_pressure = Constraint(pm['elems'], rule = elem_pressure_rule)

# CONFIGURATIONS ######################################################################


# CONFIGURATIONS & OPTIMIZATION ########################################################
    
def model_sdec(constraints, objective, pm, config='2d2s'):
    # Create a Pyomo model
    model = ConcreteModel()

    initialize_model(pm)
    # Problem-level variables

    model.recovery = Var(within=NonNegativeReals, bounds=(0,1), initialize = 1.0)
    model.separation_factor = Var(within=NonNegativeReals, bounds=(1,100), initialize = 1.0)
    model.water_recovery = Var(within=NonNegativeReals, bounds=(0,1), initialize = 0.0)
    model.p_feed = Var(within=NonNegativeReals, bounds = (0,pm['dp_max']), initialize = 0.0)
    model.power = Var(within=NonNegativeReals, bounds = (0,2000), initialize = 1500.0) # power necessary
    model.mol_power = Var(within=NonNegativeReals, bounds = (0,3000), initialize = 1500.0)

    model.final_flow = Var(within=NonNegativeReals, bounds=(0,pm['F_lim']), initialize = pm['F_feed'])
    model.final_pressure = Var(within=NonNegativeReals, bounds = (0,pm['dp_max']), initialize = 0.0)
    model.final_concentration = Var(pm['solutes'],within=NonNegativeReals, bounds = lambda b, i: pm['bounds']['C'][i], initialize = pm['c_feed_dict'])


    # Create pm['stages'] as Pyomo Blocks
    model.stages = Block(pm['stages'], rule=lambda b: stage_rule(b,pm))

    # 2D2S
    if config == '2d2s':
        def omb_0_rule(model):
            return model.stages[0].F0 == 0.5 * pm['F_feed']
        def pre_0_rule(model):
            return model.stages[0].p0 == model.p_feed
        def cmb_0_rule(model,s):
            return model.stages[0].F0 * model.stages[0].C0[s] == 0.5 * pm['F_feed'] * pm['c_feed_dict'][s]

        def omb_1_rule(model):
            return model.stages[1].F0 == model.stages[0].Fr
        def pre_1_rule(model):
            return model.stages[1].p0 == model.stages[0].pr
        def cmb_1_rule(model,s):
            return model.stages[1].F0 * model.stages[1].C0[s] == model.stages[0].Fr * model.stages[0].Cr[s]

        def omb_2_rule(model):
            return model.stages[2].F0 == 0.5 * pm['F_feed']
        def pre_2_rule(model):
            return model.stages[2].p0 == model.p_feed
        def cmb_2_rule(model,s):
            return model.stages[2].F0 * model.stages[2].C0[s] == 0.5 * pm['F_feed'] * pm['c_feed_dict'][s]

        def omb_3_rule(model):
            return model.stages[3].F0 == model.stages[2].Fr
        def pre_3_rule(model):
            return model.stages[3].p0 == model.stages[2].pr
        def cmb_3_rule(model,s):
            return model.stages[3].F0 * model.stages[3].C0[s] == model.stages[2].Fr * model.stages[2].Cr[s]

        def omb_4_rule(model):
            return model.stages[4].F0 == model.stages[1].Fr + model.stages[3].Fr
        def pre_4_rule(model):
            return model.stages[4].p0 == model.stages[1].pr
        def cmb_4_rule(model,s):
            return model.stages[4].F0 * model.stages[4].C0[s] == model.stages[1].Fr * model.stages[1].Cr[s] + model.stages[3].Fr * model.stages[3].Cr[s]

        def omb_5_rule(model):
            return model.stages[5].F0 == model.stages[4].Fr
        def pre_5_rule(model):
            return model.stages[5].p0 == model.stages[4].pr
        def cmb_5_rule(model,s):
            return model.stages[5].F0 * model.stages[5].C0[s] == model.stages[4].Fr * model.stages[4].Cr[s]

        def final_mix_rule(model):
            return model.final_flow == model.stages[5].Fr
        def final_pressure_rule(model):
            return model.final_pressure == model.stages[5].pr
        def final_component_rule(model,s):
            return model.final_flow * model.final_concentration[s] == model.stages[5].Fr * model.stages[5].Cr[s]

    #3D
    elif config == '3d':
        def omb_0_rule(model):
            return model.stages[0].F0 == 0.5 * pm['F_feed']
        def pre_0_rule(model):
            return model.stages[0].p0 == model.p_feed
        def cmb_0_rule(model,s):
            return model.stages[0].F0 * model.stages[0].C0[s] == 0.5 * pm['F_feed'] * pm['c_feed_dict'][s]

        def omb_1_rule(model):
            return model.stages[1].F0 == model.stages[0].Fr
        def pre_1_rule(model):
            return model.stages[1].p0 == model.stages[0].pr
        def cmb_1_rule(model,s):
            return model.stages[1].F0 * model.stages[1].C0[s] == model.stages[0].Fr * model.stages[0].Cr[s]

        def omb_2_rule(model):
            return model.stages[2].F0 == 0.5 * pm['F_feed']
        def pre_2_rule(model):
            return model.stages[2].p0 == model.p_feed
        def cmb_2_rule(model,s):
            return model.stages[2].F0 * model.stages[2].C0[s] == 0.5 * pm['F_feed'] * pm['c_feed_dict'][s]

        def omb_3_rule(model):
            return model.stages[3].F0 == model.stages[2].Fr
        def pre_3_rule(model):
            return model.stages[3].p0 == model.stages[2].pr
        def cmb_3_rule(model,s):
            return model.stages[3].F0 * model.stages[3].C0[s] == model.stages[2].Fr * model.stages[2].Cr[s]

        def omb_4_rule(model):
            return model.stages[4].F0 == model.stages[1].Fr
        def pre_4_rule(model):
            return model.stages[4].p0 == model.stages[1].pr
        def cmb_4_rule(model,s):
            return model.stages[4].F0 * model.stages[4].C0[s] == model.stages[1].Fr * model.stages[1].Cr[s]

        def omb_5_rule(model):
            return model.stages[5].F0 == model.stages[3].Fr
        def pre_5_rule(model):
            return model.stages[5].p0 == model.stages[3].pr
        def cmb_5_rule(model,s):
            return model.stages[5].F0 * model.stages[5].C0[s] == model.stages[3].Fr * model.stages[3].Cr[s]

        def final_mix_rule(model):
            return model.final_flow == model.stages[4].Fr + model.stages[5].Fr
        def final_pressure_rule(model):
            return model.final_pressure == model.stages[4].pr
        def final_component_rule(model,s):
            return model.final_flow * model.final_concentration[s] == model.stages[4].Fr * model.stages[4].Cr[s] + model.stages[5].Fr * model.stages[5].Cr[s]
    else:
        print('Unknown config')
        return

    model.omb_0 = Constraint(rule=omb_0_rule)
    model.cmb_0 = Constraint(pm['solutes'],rule=cmb_0_rule)
    model.omb_1 = Constraint(rule=omb_1_rule)
    model.cmb_1 = Constraint(pm['solutes'],rule=cmb_1_rule)
    model.omb_2 = Constraint(rule=omb_2_rule)
    model.cmb_2 = Constraint(pm['solutes'],rule=cmb_2_rule)
    model.omb_3 = Constraint(rule=omb_3_rule)
    model.cmb_3 = Constraint(pm['solutes'],rule=cmb_3_rule)
    model.omb_4 = Constraint(rule=omb_4_rule)
    model.cmb_4 = Constraint(pm['solutes'],rule=cmb_4_rule)
    model.omb_5 = Constraint(rule=omb_5_rule)
    model.cmb_5 = Constraint(pm['solutes'],rule=cmb_5_rule)

    model.pre_0 = Constraint(rule=pre_0_rule)
    model.pre_1 = Constraint(rule=pre_1_rule)
    model.pre_2 = Constraint(rule=pre_2_rule)
    model.pre_3 = Constraint(rule=pre_3_rule)
    model.pre_4 = Constraint(rule=pre_4_rule)
    model.pre_5 = Constraint(rule=pre_5_rule)

    model.final_mix_constr = Constraint(rule=final_mix_rule)
    model.final_pressure_constr = Constraint(rule=final_pressure_rule)
    model.final_component_constr = Constraint(pm['solutes'],rule=final_component_rule)


    # Performance metrics
    def recovery_rule(model):
        return (model.final_flow * (model.final_concentration[0]+model.final_concentration[1])) / (pm['F_feed'] * (pm['c_feed_dict'][0]+pm['c_feed_dict'][1])) == model.recovery

    def water_recovery_rule(model):
        return 1 - (model.final_flow/(pm['F_feed'])) == model.water_recovery

    def separation_factor_rule(model):
        return model.separation_factor == ((model.final_concentration[0]+model.final_concentration[1]) / (model.final_concentration[2]+model.final_concentration[3])) / ((pm['c_feed_dict'][0]+pm['c_feed_dict'][1]) / (pm['c_feed_dict'][2]+pm['c_feed_dict'][3]))

    model.recovery_constr = Constraint(rule=recovery_rule)
    model.water_recovery_constr = Constraint(rule=water_recovery_rule)
    model.separation_factor_constr = Constraint(rule=separation_factor_rule)


    # Constraints on pressure exchanger and power
    def power_rule(model):
        return model.power == (1/pm['pump_eff']) * pm['F_feed'] * model.p_feed

    def molar_power_rule(model):
        return model.mol_power == model.power / model.recovery

    model.power_constr = Constraint(rule=power_rule)
    model.molar_power_rule = Constraint(rule=molar_power_rule)


    # Multi-objective constraints
    if 'recovery' in constraints:
        def mo_rec(model):
            return model.recovery >= constraints['recovery']

        model.mo_rec = Constraint(rule=mo_rec)

    if 'separation_factor' in constraints:
        def mo_sf_rule(model):
            return model.separation_factor >= constraints['separation_factor']

        model.mo_sf = Constraint(rule=mo_sf_rule)
    
    if 'molar_power' in constraints:
        def mo_sp_rule(model):
            return model.mol_power <= constraints['molar_power']

        model.mo_sp = Constraint(rule=mo_sp_rule)


    # Objective
    if objective =='separation_factor':
        model.obj = Objective(expr=((model.final_concentration[2]+model.final_concentration[3]) / (sum(model.final_concentration[i] for i in range(4)))))
    elif objective == 'recovery':
        model.obj = Objective(expr=(1-model.recovery))
    elif objective =='molar_power':
        model.obj = Objective(expr=(model.mol_power))

    return model
    



def model_sdec_x(constraints, objective, pm, config='2d2s'):
    # Create a Pyomo model
    model = ConcreteModel()

    initialize_model(pm)

    # Problem-level variables

    model.recovery = Var(within=NonNegativeReals, bounds=(0,1), initialize = 0.0)
    model.separation_factor = Var(within=NonNegativeReals, bounds=(1,100), initialize = 1.0)
    model.water_recovery = Var(within=NonNegativeReals, bounds=(0,1), initialize = 0.0)
    model.p_pump = Var(within=NonNegativeReals, bounds = (0,pm['dp_max']), initialize = 0.0) # pressure generated by pump
    model.p_pex = Var(within=NonNegativeReals, bounds = (0,pm['dp_max']), initialize = 0.0) # pressure generated by pressure exchanger after the pump
    model.power = Var(within=NonNegativeReals, bounds = (0,2000), initialize = 1500.0) # power necessary
    model.mol_power = Var(within=NonNegativeReals, bounds = (0,3000), initialize = 1500.0)

    model.final_flow = Var(within=NonNegativeReals, bounds=(0,pm['F_lim']), initialize = pm['F_feed'])
    model.final_pressure = Var(within=NonNegativeReals, bounds = (0,pm['dp_max']), initialize = 0.0)
    model.final_concentration = Var(pm['solutes'],within=NonNegativeReals, bounds = lambda b, i: pm['bounds']['C'][i], initialize = pm['c_feed_dict'])


    # Create pm['stages'] as Pyomo Blocks
    model.stages = Block(pm['stages'], rule=lambda b: stage_rule(b,pm))


    # 2D2S
    if config == '2d2s':
        def omb_0_rule(model):
            return model.stages[0].F0 == 0.5 * pm['F_feed']
        def pre_0_rule(model):
            return model.stages[0].p0 == model.p_pex
        def cmb_0_rule(model,s):
            return model.stages[0].F0 * model.stages[0].C0[s] == 0.5 * pm['F_feed'] * pm['c_feed_dict'][s]

        def omb_1_rule(model):
            return model.stages[1].F0 == model.stages[0].Fr
        def pre_1_rule(model):
            return model.stages[1].p0 == model.stages[0].pr
        def cmb_1_rule(model,s):
            return model.stages[1].F0 * model.stages[1].C0[s] == model.stages[0].Fr * model.stages[0].Cr[s]

        def omb_2_rule(model):
            return model.stages[2].F0 == 0.5 * pm['F_feed']
        def pre_2_rule(model):
            return model.stages[2].p0 == model.p_pex
        def cmb_2_rule(model,s):
            return model.stages[2].F0 * model.stages[2].C0[s] == 0.5 * pm['F_feed'] * pm['c_feed_dict'][s]

        def omb_3_rule(model):
            return model.stages[3].F0 == model.stages[2].Fr
        def pre_3_rule(model):
            return model.stages[3].p0 == model.stages[2].pr
        def cmb_3_rule(model,s):
            return model.stages[3].F0 * model.stages[3].C0[s] == model.stages[2].Fr * model.stages[2].Cr[s]

        def omb_4_rule(model):
            return model.stages[4].F0 == model.stages[1].Fr + model.stages[3].Fr
        def pre_4_rule(model):
            return model.stages[4].p0 == model.stages[1].pr
        def cmb_4_rule(model,s):
            return model.stages[4].F0 * model.stages[4].C0[s] == model.stages[1].Fr * model.stages[1].Cr[s] + model.stages[3].Fr * model.stages[3].Cr[s]

        def omb_5_rule(model):
            return model.stages[5].F0 == model.stages[4].Fr
        def pre_5_rule(model):
            return model.stages[5].p0 == model.stages[4].pr
        def cmb_5_rule(model,s):
            return model.stages[5].F0 * model.stages[5].C0[s] == model.stages[4].Fr * model.stages[4].Cr[s]

        def final_mix_rule(model):
            return model.final_flow == model.stages[5].Fr
        def final_pressure_rule(model):
            return model.final_pressure == model.stages[5].pr
        def final_component_rule(model,s):
            return model.final_flow * model.final_concentration[s] == model.stages[5].Fr * model.stages[5].Cr[s]
    else:
        print('Unknown config')
        return

    model.omb_0 = Constraint(rule=omb_0_rule)
    model.cmb_0 = Constraint(pm['solutes'],rule=cmb_0_rule)
    model.omb_1 = Constraint(rule=omb_1_rule)
    model.cmb_1 = Constraint(pm['solutes'],rule=cmb_1_rule)
    model.omb_2 = Constraint(rule=omb_2_rule)
    model.cmb_2 = Constraint(pm['solutes'],rule=cmb_2_rule)
    model.omb_3 = Constraint(rule=omb_3_rule)
    model.cmb_3 = Constraint(pm['solutes'],rule=cmb_3_rule)
    model.omb_4 = Constraint(rule=omb_4_rule)
    model.cmb_4 = Constraint(pm['solutes'],rule=cmb_4_rule)
    model.omb_5 = Constraint(rule=omb_5_rule)
    model.cmb_5 = Constraint(pm['solutes'],rule=cmb_5_rule)

    model.pre_0 = Constraint(rule=pre_0_rule)
    model.pre_1 = Constraint(rule=pre_1_rule)
    model.pre_2 = Constraint(rule=pre_2_rule)
    model.pre_3 = Constraint(rule=pre_3_rule)
    model.pre_4 = Constraint(rule=pre_4_rule)
    model.pre_5 = Constraint(rule=pre_5_rule)

    model.final_mix_constr = Constraint(rule=final_mix_rule)
    model.final_pressure_constr = Constraint(rule=final_pressure_rule)
    model.final_component_constr = Constraint(pm['solutes'],rule=final_component_rule)


    # Performance metrics
    def recovery_rule(model):
        return (model.final_flow * (model.final_concentration[0]+model.final_concentration[1])) / (pm['F_feed'] * (pm['c_feed_dict'][0]+pm['c_feed_dict'][1])) == model.recovery

    def water_recovery_rule(model):
        return 1 - (model.final_flow/(pm['F_feed'])) == model.water_recovery

    def separation_factor_rule(model):
        return model.separation_factor == ((model.final_concentration[0]+model.final_concentration[1]) / (model.final_concentration[2]+model.final_concentration[3])) / ((pm['c_feed_dict'][0]+pm['c_feed_dict'][1]) / (pm['c_feed_dict'][2]+pm['c_feed_dict'][3]))

    model.recovery_constr = Constraint(rule=recovery_rule)
    model.water_recovery_constr = Constraint(rule=water_recovery_rule)
    model.separation_factor_constr = Constraint(rule=separation_factor_rule)


    # Constraints on pressure exchanger and power
    def power_rule(model):
        return model.power == (1/pm['pump_eff']) * pm['F_feed'] * model.p_pump

    def molar_power_rule(model):
        return model.mol_power == model.power / model.recovery

    def pex_rule(model):
        return model.p_pex == (pm['pex_eff'] * model.final_flow * model.final_pressure + pm['F_feed'] * model.p_pump)/pm['F_feed']

    model.power_constr = Constraint(rule=power_rule)
    model.molar_power_rule = Constraint(rule=molar_power_rule)
    model.pex_constraint = Constraint(rule=pex_rule)


    # Multi-objective constraints
    if 'recovery' in constraints:
        def mo_rec(model):
            return model.recovery >= constraints['recovery']

        model.mo_rec = Constraint(rule=mo_rec)

    if 'separation_factor' in constraints:
        def mo_sf_rule(model):
            return model.separation_factor >= constraints['separation_factor']

        model.mo_sf = Constraint(rule=mo_sf_rule)
    
    if 'molar_power' in constraints:
        def mo_sp_rule(model):
            return model.mol_power <= constraints['molar_power']

        model.mo_sp = Constraint(rule=mo_sp_rule)


    # Objective
    if objective =='separation_factor':
        model.obj = Objective(expr=((model.final_concentration[2]+model.final_concentration[3]) / (sum(model.final_concentration[i] for i in range(4)))))
    elif objective == 'recovery':
        model.obj = Objective(expr=(1-model.recovery))
    elif objective =='molar_power':
        model.obj = Objective(expr=(model.mol_power))

    return model
    


def model_sdec_d(constraints, objective, pm, config = '2d2s'):
    # Create a Pyomo model
    model = ConcreteModel()
    initialize_model(pm)

    # Problem-level variables

    model.recovery = Var(within=NonNegativeReals, bounds=(0,1), initialize = 0.0)
    model.separation_factor = Var(within=NonNegativeReals, bounds=(1,100), initialize = 1.0)
    model.water_recovery = Var(within=NonNegativeReals, bounds=(0,1), initialize = 0.0)
    model.p_feed = Var(within=NonNegativeReals, bounds = (0,pm['dp_max']), initialize = 0.0) # pressure generated by pump
    model.power = Var(within=NonNegativeReals, bounds = (0,2000), initialize = 1500.0) # power necessary
    model.mol_power = Var(within=NonNegativeReals, bounds = (0,3000), initialize = 1500.0)
    model.F_dilution = Var(pm['stages'],within=NonNegativeReals, bounds=(0,pm['F_dil_feed']), initialize = 0.0)
    model.power_dilution = Var(pm['stages'],within=NonNegativeReals, bounds = (0,2000), initialize = 1500.0)

    model.final_flow = Var(within=NonNegativeReals, bounds=(0,pm['F_lim']), initialize = pm['F_feed'])
    model.final_pressure = Var(within=NonNegativeReals, bounds = (0,pm['dp_max']), initialize = 0.0)
    model.final_concentration = Var(pm['solutes'],within=NonNegativeReals, bounds = lambda b, i: pm['bounds']['C'][i], initialize = pm['c_feed_dict'])


    # Create pm['stages'] as Pyomo Blocks
    model.stages = Block(pm['stages'], rule=lambda b: stage_rule(b,pm))

    if config == '2d2s':
        def omb_0_rule(model):
            return model.stages[0].F0 == 0.5 * pm['F_feed'] + model.F_dilution[0]
        def pre_0_rule(model):
            return model.stages[0].p0 == model.p_feed
        def cmb_0_rule(model,s):
            return model.stages[0].F0 * model.stages[0].C0[s] == 0.5 * pm['F_feed'] * pm['c_feed_dict'][s]

        def omb_1_rule(model):
            return model.stages[1].F0 == model.stages[0].Fr + model.F_dilution[1]
        def pre_1_rule(model):
            return model.stages[1].p0 == model.stages[0].pr
        def cmb_1_rule(model,s):
            return model.stages[1].F0 * model.stages[1].C0[s] == model.stages[0].Fr * model.stages[0].Cr[s]

        def omb_2_rule(model):
            return model.stages[2].F0 == 0.5 * pm['F_feed'] + model.F_dilution[2]
        def pre_2_rule(model):
            return model.stages[2].p0 == model.p_feed
        def cmb_2_rule(model,s):
            return model.stages[2].F0 * model.stages[2].C0[s] == 0.5 * pm['F_feed'] * pm['c_feed_dict'][s]

        def omb_3_rule(model):
            return model.stages[3].F0 == model.stages[2].Fr + model.F_dilution[3]
        def pre_3_rule(model):
            return model.stages[3].p0 == model.stages[2].pr
        def cmb_3_rule(model,s):
            return model.stages[3].F0 * model.stages[3].C0[s] == model.stages[2].Fr * model.stages[2].Cr[s]

        def omb_4_rule(model):
            return model.stages[4].F0 == model.stages[1].Fr + model.stages[3].Fr + model.F_dilution[4]
        def pre_4_rule(model):
            return model.stages[4].p0 == model.stages[1].pr
        def cmb_4_rule(model,s):
            return model.stages[4].F0 * model.stages[4].C0[s] == model.stages[1].Fr * model.stages[1].Cr[s] + model.stages[3].Fr * model.stages[3].Cr[s]

        def omb_5_rule(model):
            return model.stages[5].F0 == model.stages[4].Fr + model.F_dilution[5]
        def pre_5_rule(model):
            return model.stages[5].p0 == model.stages[4].pr
        def cmb_5_rule(model,s):
            return model.stages[5].F0 * model.stages[5].C0[s] == model.stages[4].Fr * model.stages[4].Cr[s]
        
        def final_mix_rule(model):
            return model.final_flow == model.stages[5].Fr
        def final_pressure_rule(model):
            return model.final_pressure == model.stages[5].pr
        def final_component_rule(model,s):
            return model.final_flow * model.final_concentration[s] == model.stages[5].Fr * model.stages[5].Cr[s]
    else:
        print('Unknown config')
        return

    model.omb_0 = Constraint(rule=omb_0_rule)
    model.cmb_0 = Constraint(pm['solutes'],rule=cmb_0_rule)
    model.omb_1 = Constraint(rule=omb_1_rule)
    model.cmb_1 = Constraint(pm['solutes'],rule=cmb_1_rule)
    model.omb_2 = Constraint(rule=omb_2_rule)
    model.cmb_2 = Constraint(pm['solutes'],rule=cmb_2_rule)
    model.omb_3 = Constraint(rule=omb_3_rule)
    model.cmb_3 = Constraint(pm['solutes'],rule=cmb_3_rule)
    model.omb_4 = Constraint(rule=omb_4_rule)
    model.cmb_4 = Constraint(pm['solutes'],rule=cmb_4_rule)
    model.omb_5 = Constraint(rule=omb_5_rule)
    model.cmb_5 = Constraint(pm['solutes'],rule=cmb_5_rule)

    model.pre_0 = Constraint(rule=pre_0_rule)
    model.pre_1 = Constraint(rule=pre_1_rule)
    model.pre_2 = Constraint(rule=pre_2_rule)
    model.pre_3 = Constraint(rule=pre_3_rule)
    model.pre_4 = Constraint(rule=pre_4_rule)
    model.pre_5 = Constraint(rule=pre_5_rule)


    model.final_mix_constr = Constraint(rule=final_mix_rule)
    model.final_pressure_constr = Constraint(rule=final_pressure_rule)
    model.final_component_constr = Constraint(pm['solutes'],rule=final_component_rule)


    # Performance metrics
    def recovery_rule(model):
        return (model.final_flow * (model.final_concentration[0]+model.final_concentration[1])) / (pm['F_feed'] * (pm['c_feed_dict'][0]+pm['c_feed_dict'][1])) == model.recovery

    def water_recovery_rule(model):
        return 1 - (model.final_flow/(pm['F_feed']+sum(model.F_dilution[i] for i in pm['stages']))) == model.water_recovery

    def separation_factor_rule(model):
        return model.separation_factor == ((model.final_concentration[0]+model.final_concentration[1]) / (model.final_concentration[2]+model.final_concentration[3])) / ((pm['c_feed_dict'][0]+pm['c_feed_dict'][1]) / (pm['c_feed_dict'][2]+pm['c_feed_dict'][3]))

    model.recovery_constr = Constraint(rule=recovery_rule)
    model.water_recovery_constr = Constraint(rule=water_recovery_rule)
    model.separation_factor_constr = Constraint(rule=separation_factor_rule)


    # Constraint on dilution
    def dilution_rule(model):
        return sum(model.F_dilution[i] for i in pm['stages']) <= pm['F_dil_feed']

    model.dilution_constr = Constraint(rule=dilution_rule)


    # Constraints on pressure exchanger and power
    def power_rule(model):
        return model.power == (1/pm['pump_eff']) * pm['F_feed'] * model.p_feed + sum(model.power_dilution[i] for i in pm['stages'])

    def dilution_power_rule(model,st):
        return model.power_dilution[st] == (1/pm['pump_eff']) * model.F_dilution[st] * model.stages[st].p0

    def molar_power_rule(model):
        return model.mol_power == model.power / model.recovery


    model.power_constr = Constraint(rule=power_rule)
    model.dilution_power_constr = Constraint(pm['stages'],rule=dilution_power_rule)
    model.molar_power_rule = Constraint(rule=molar_power_rule)

    # Multi-objective constraints
    if 'recovery' in constraints:
        def mo_rec(model):
            return model.recovery >= constraints['recovery']

        model.mo_rec = Constraint(rule=mo_rec)

    if 'separation_factor' in constraints:
        def mo_sf_rule(model):
            return model.separation_factor >= constraints['separation_factor']

        model.mo_sf = Constraint(rule=mo_sf_rule)
    
    if 'molar_power' in constraints:
        def mo_sp_rule(model):
            return model.mol_power <= constraints['molar_power']

        model.mo_sp = Constraint(rule=mo_sp_rule)


    # Objective
    if objective =='separation_factor':
        model.obj = Objective(expr=((model.final_concentration[2]+model.final_concentration[3]) / (sum(model.final_concentration[i] for i in range(4)))))
    elif objective == 'recovery':
        model.obj = Objective(expr=(1-model.recovery))
    elif objective =='molar_power':
        model.obj = Objective(expr=(model.mol_power))

    return model


def model_sdec_r(constraints, objective, pm, config = '2d2s'):
    # Create a Pyomo model
    model = ConcreteModel()
    initialize_model(pm)
    # Problem-level variables

    model.recovery = Var(within=NonNegativeReals, bounds=(0,1), initialize = 0.0)
    model.separation_factor = Var(within=NonNegativeReals, bounds=(1,100), initialize = 1.0)
    model.water_recovery = Var(within=NonNegativeReals, bounds=(0,1), initialize = 0.0)
    model.p_feed = Var(within=NonNegativeReals, bounds = (0,pm['dp_max']), initialize = 0.0) # pressure generated by pump
    model.power = Var(within=NonNegativeReals, bounds = (0,2000), initialize = 1500.0) # power necessary
    model.mol_power = Var(within=NonNegativeReals, bounds = (0,3000), initialize = 1500.0)
    model.omega = Var(pm['stages'],within=NonNegativeReals, bounds=(0,1), initialize = 1)
    model.recirc_power = Var(pm['stages'],within=NonNegativeReals, bounds=(0,2000), initialize = 0.0)

    model.final_flow = Var(within=NonNegativeReals, bounds=(0,pm['F_lim']), initialize = pm['F_feed'])
    model.final_pressure = Var(within=NonNegativeReals, bounds = (0,pm['dp_max']), initialize = 0.0)
    model.final_concentration = Var(pm['solutes'],within=NonNegativeReals, bounds = lambda b, i: pm['bounds']['C'][i], initialize = pm['c_feed_dict'])


    # Create pm['stages'] as Pyomo Blocks
    model.stages = Block(pm['stages'], rule=lambda b: stage_rule(b,pm))

    # Define mass balance and pressure constraints for each membrane element in each stage
    if config == '2d2s':
        def omb_0_rule(model):
            return model.stages[0].F0 == 0.5 * pm['F_feed'] + model.omega[1] * model.stages[1].Fp
        def pre_0_rule(model):
            return model.stages[0].p0 == model.p_feed
        def cmb_0_rule(model,s):
            return model.stages[0].F0 * model.stages[0].C0[s] == 0.5 * pm['F_feed'] * pm['c_feed_dict'][s] + model.omega[1] * model.stages[1].Fp * model.stages[1].Cp[s]

        def omb_1_rule(model):
            return model.stages[1].F0 == model.stages[0].Fr + 0.5 * model.omega[4] * model.stages[4].Fp
        def pre_1_rule(model):
            return model.stages[1].p0 == model.stages[0].pr
        def cmb_1_rule(model,s):
            return model.stages[1].F0 * model.stages[1].C0[s] == model.stages[0].Fr * model.stages[0].Cr[s] + 0.5 * model.omega[4] * model.stages[4].Fp * model.stages[4].Cp[s]
        def recirc_1_rule(model):
            return model.recirc_power[1] == (1/pm['pump_eff']) * model.stages[1].Fp * model.omega[1] * (model.stages[0].p0 - model.stages[1].pp)

        def omb_2_rule(model):
            return model.stages[2].F0 == 0.5 * pm['F_feed'] + model.omega[3] * model.stages[3].Fp
        def pre_2_rule(model):
            return model.stages[2].p0 == model.p_feed
        def cmb_2_rule(model,s):
            return model.stages[2].F0 * model.stages[2].C0[s] == 0.5 * pm['F_feed'] * pm['c_feed_dict'][s] + model.omega[3] * model.stages[3].Fp * model.stages[3].Cp[s]

        def omb_3_rule(model):
            return model.stages[3].F0 == model.stages[2].Fr + 0.5 * model.omega[4] * model.stages[4].Fp
        def pre_3_rule(model):
            return model.stages[3].p0 == model.stages[2].pr
        def cmb_3_rule(model,s):
            return model.stages[3].F0 * model.stages[3].C0[s] == model.stages[2].Fr * model.stages[2].Cr[s] + 0.5 * model.omega[4] * model.stages[4].Fp * model.stages[4].Cp[s]
        def recirc_3_rule(model):
            return model.recirc_power[3] == (1/pm['pump_eff']) * model.stages[3].Fp * model.omega[3] * (model.stages[2].p0 - model.stages[3].pp)

        def omb_4_rule(model):
            return model.stages[4].F0 == model.stages[1].Fr + model.stages[3].Fr + model.omega[5] * model.stages[5].Fp
        def pre_4_rule(model):
            return model.stages[4].p0 == model.stages[1].pr
        def cmb_4_rule(model,s):
            return model.stages[4].F0 * model.stages[4].C0[s] == model.stages[1].Fr * model.stages[1].Cr[s] + model.stages[3].Fr * model.stages[3].Cr[s] + model.omega[5] * model.stages[5].Fp * model.stages[5].Cp[s]
        def recirc_4_rule(model):
            return model.recirc_power[4] == 0.5 * (1/pm['pump_eff']) * model.stages[4].Fp * model.omega[4] * (model.stages[1].p0 - model.stages[4].pp) + 0.5 * (1/pm['pump_eff']) * model.stages[4].Fp * model.omega[4] * (model.stages[3].p0 - model.stages[4].pp)

        def omb_5_rule(model):
            return model.stages[5].F0 == model.stages[4].Fr
        def pre_5_rule(model):
            return model.stages[5].p0 == model.stages[4].pr
        def cmb_5_rule(model,s):
            return model.stages[5].F0 * model.stages[5].C0[s] == model.stages[4].Fr * model.stages[4].Cr[s]
        def recirc_5_rule(model):
            return model.recirc_power[5] == (1/pm['pump_eff']) * model.stages[5].Fp * model.omega[5] * (model.stages[4].p0 - model.stages[5].pp)
        
        def final_mix_rule(model):
            return model.final_flow == model.stages[5].Fr
        def final_pressure_rule(model):
            return model.final_pressure == model.stages[5].pr
        def final_component_rule(model,s):
            return model.final_flow * model.final_concentration[s] == model.stages[5].Fr * model.stages[5].Cr[s]
    else:
        print('Unknown config')
        return
    
    # Assignment
    model.omb_0 = Constraint(rule=omb_0_rule)
    model.cmb_0 = Constraint(pm['solutes'],rule=cmb_0_rule)
    model.omb_1 = Constraint(rule=omb_1_rule)
    model.cmb_1 = Constraint(pm['solutes'],rule=cmb_1_rule)
    model.omb_2 = Constraint(rule=omb_2_rule)
    model.cmb_2 = Constraint(pm['solutes'],rule=cmb_2_rule)
    model.omb_3 = Constraint(rule=omb_3_rule)
    model.cmb_3 = Constraint(pm['solutes'],rule=cmb_3_rule)
    model.omb_4 = Constraint(rule=omb_4_rule)
    model.cmb_4 = Constraint(pm['solutes'],rule=cmb_4_rule)
    model.omb_5 = Constraint(rule=omb_5_rule)
    model.cmb_5 = Constraint(pm['solutes'],rule=cmb_5_rule)

    model.pre_0 = Constraint(rule=pre_0_rule)
    model.pre_1 = Constraint(rule=pre_1_rule)
    model.pre_2 = Constraint(rule=pre_2_rule)
    model.pre_3 = Constraint(rule=pre_3_rule)
    model.pre_4 = Constraint(rule=pre_4_rule)
    model.pre_5 = Constraint(rule=pre_5_rule)

    model.recirc_const1 = Constraint(rule=recirc_1_rule)
    model.recirc_const3 = Constraint(rule=recirc_3_rule)
    model.recirc_const4 = Constraint(rule=recirc_4_rule)
    model.recirc_const5 = Constraint(rule=recirc_5_rule)


    model.final_mix_constr = Constraint(rule=final_mix_rule)
    model.final_pressure_constr = Constraint(rule=final_pressure_rule)
    model.final_component_constr = Constraint(pm['solutes'],rule=final_component_rule)


    # Performance metrics
    def recovery_rule(model):
        return (model.final_flow * (model.final_concentration[0]+model.final_concentration[1])) / (pm['F_feed'] * (pm['c_feed_dict'][0]+pm['c_feed_dict'][1])) == model.recovery

    def water_recovery_rule(model):
        return 1 - (model.final_flow/(pm['F_feed'])) == model.water_recovery

    def separation_factor_rule(model):
        return model.separation_factor == ((model.final_concentration[0]+model.final_concentration[1]) / (model.final_concentration[2]+model.final_concentration[3])) / ((pm['c_feed_dict'][0]+pm['c_feed_dict'][1]) / (pm['c_feed_dict'][2]+pm['c_feed_dict'][3]))

    model.recovery_constr = Constraint(rule=recovery_rule)
    model.water_recovery_constr = Constraint(rule=water_recovery_rule)
    model.separation_factor_constr = Constraint(rule=separation_factor_rule)



    # Constraints on pressure exchanger and power
    def power_rule(model):
        return model.power == (1/pm['pump_eff']) * pm['F_feed'] * model.p_feed + model.recirc_power[1] +  model.recirc_power[3] +  model.recirc_power[4] +  model.recirc_power[5]

    def molar_power_rule(model):
        return model.mol_power == model.power / model.recovery


    model.power_constr = Constraint(rule=power_rule)
    model.molar_power_rule = Constraint(rule=molar_power_rule)

    # Multi-objective constraints
    if 'recovery' in constraints:
        def mo_rec(model):
            return model.recovery >= constraints['recovery']

        model.mo_rec = Constraint(rule=mo_rec)

    if 'separation_factor' in constraints:
        def mo_sf_rule(model):
            return model.separation_factor >= constraints['separation_factor']

        model.mo_sf = Constraint(rule=mo_sf_rule)
    
    if 'molar_power' in constraints:
        def mo_sp_rule(model):
            return model.mol_power <= constraints['molar_power']

        model.mo_sp = Constraint(rule=mo_sp_rule)


    # Objective
    if objective =='separation_factor':
        model.obj = Objective(expr=((model.final_concentration[2]+model.final_concentration[3]) / (sum(model.final_concentration[i] for i in range(4)))))
    elif objective == 'recovery':
        model.obj = Objective(expr=(1-model.recovery))
    elif objective =='molar_power':
        model.obj = Objective(expr=(model.mol_power))

    return model



def model_sdec_xd(constraints, objective, pm, config = '2d2s'):
    # Create a Pyomo model
    model = ConcreteModel()
    initialize_model(pm)

    # Problem-level variables

    model.recovery = Var(within=NonNegativeReals, bounds=(0,1), initialize = 0.0)
    model.separation_factor = Var(within=NonNegativeReals, bounds=(1,100), initialize = 1.0)
    model.water_recovery = Var(within=NonNegativeReals, bounds=(0,1), initialize = 0.0)
    model.p_pump = Var(within=NonNegativeReals, bounds = (0,pm['dp_max']), initialize = 0.0) # pressure generated by pump
    model.p_pex = Var(within=NonNegativeReals, bounds = (0,pm['dp_max']), initialize = 0.0) # pressure generated by pressure exchanger after the pump
    model.power = Var(within=NonNegativeReals, bounds = (0,2000), initialize = 1500.0) # power necessary
    model.mol_power = Var(within=NonNegativeReals, bounds = (0,3000), initialize = 1500.0)
    model.F_dilution = Var(pm['stages'],within=NonNegativeReals, bounds=(0,pm['F_dil_feed']), initialize = 0.0)
    model.power_dilution = Var(pm['stages'],within=NonNegativeReals, bounds = (0,2000), initialize = 1500.0)

    model.final_flow = Var(within=NonNegativeReals, bounds=(0,pm['F_lim']), initialize = pm['F_feed'])
    model.final_pressure = Var(within=NonNegativeReals, bounds = (0,pm['dp_max']), initialize = 0.0)
    model.final_concentration = Var(pm['solutes'],within=NonNegativeReals, bounds = lambda b, i: pm['bounds']['C'][i], initialize = pm['c_feed_dict'])


    # Create pm['stages'] as Pyomo Blocks
    model.stages = Block(pm['stages'], rule=lambda b: stage_rule(b,pm))

    if config == '2d2s':
        def omb_0_rule(model):
            return model.stages[0].F0 == 0.5 * pm['F_feed'] + model.F_dilution[0]
        def pre_0_rule(model):
            return model.stages[0].p0 == model.p_pex
        def cmb_0_rule(model,s):
            return model.stages[0].F0 * model.stages[0].C0[s] == 0.5 * pm['F_feed'] * pm['c_feed_dict'][s]

        def omb_1_rule(model):
            return model.stages[1].F0 == model.stages[0].Fr + model.F_dilution[1]
        def pre_1_rule(model):
            return model.stages[1].p0 == model.stages[0].pr
        def cmb_1_rule(model,s):
            return model.stages[1].F0 * model.stages[1].C0[s] == model.stages[0].Fr * model.stages[0].Cr[s]

        def omb_2_rule(model):
            return model.stages[2].F0 == 0.5 * pm['F_feed'] + model.F_dilution[2]
        def pre_2_rule(model):
            return model.stages[2].p0 == model.p_pex
        def cmb_2_rule(model,s):
            return model.stages[2].F0 * model.stages[2].C0[s] == 0.5 * pm['F_feed'] * pm['c_feed_dict'][s]

        def omb_3_rule(model):
            return model.stages[3].F0 == model.stages[2].Fr + model.F_dilution[3]
        def pre_3_rule(model):
            return model.stages[3].p0 == model.stages[2].pr
        def cmb_3_rule(model,s):
            return model.stages[3].F0 * model.stages[3].C0[s] == model.stages[2].Fr * model.stages[2].Cr[s]

        def omb_4_rule(model):
            return model.stages[4].F0 == model.stages[1].Fr + model.stages[3].Fr + model.F_dilution[4]
        def pre_4_rule(model):
            return model.stages[4].p0 == model.stages[1].pr
        def cmb_4_rule(model,s):
            return model.stages[4].F0 * model.stages[4].C0[s] == model.stages[1].Fr * model.stages[1].Cr[s] + model.stages[3].Fr * model.stages[3].Cr[s]

        def omb_5_rule(model):
            return model.stages[5].F0 == model.stages[4].Fr + model.F_dilution[5]
        def pre_5_rule(model):
            return model.stages[5].p0 == model.stages[4].pr
        def cmb_5_rule(model,s):
            return model.stages[5].F0 * model.stages[5].C0[s] == model.stages[4].Fr * model.stages[4].Cr[s]
        
        def final_mix_rule(model):
            return model.final_flow == model.stages[5].Fr
        def final_pressure_rule(model):
            return model.final_pressure == model.stages[5].pr
        def final_component_rule(model,s):
            return model.final_flow * model.final_concentration[s] == model.stages[5].Fr * model.stages[5].Cr[s]
    else:
        print('Unknown config')
        return

    model.omb_0 = Constraint(rule=omb_0_rule)
    model.cmb_0 = Constraint(pm['solutes'],rule=cmb_0_rule)
    model.omb_1 = Constraint(rule=omb_1_rule)
    model.cmb_1 = Constraint(pm['solutes'],rule=cmb_1_rule)
    model.omb_2 = Constraint(rule=omb_2_rule)
    model.cmb_2 = Constraint(pm['solutes'],rule=cmb_2_rule)
    model.omb_3 = Constraint(rule=omb_3_rule)
    model.cmb_3 = Constraint(pm['solutes'],rule=cmb_3_rule)
    model.omb_4 = Constraint(rule=omb_4_rule)
    model.cmb_4 = Constraint(pm['solutes'],rule=cmb_4_rule)
    model.omb_5 = Constraint(rule=omb_5_rule)
    model.cmb_5 = Constraint(pm['solutes'],rule=cmb_5_rule)

    model.pre_0 = Constraint(rule=pre_0_rule)
    model.pre_1 = Constraint(rule=pre_1_rule)
    model.pre_2 = Constraint(rule=pre_2_rule)
    model.pre_3 = Constraint(rule=pre_3_rule)
    model.pre_4 = Constraint(rule=pre_4_rule)
    model.pre_5 = Constraint(rule=pre_5_rule)


    model.final_mix_constr = Constraint(rule=final_mix_rule)
    model.final_pressure_constr = Constraint(rule=final_pressure_rule)
    model.final_component_constr = Constraint(pm['solutes'],rule=final_component_rule)


    # Performance metrics
    def recovery_rule(model):
        return (model.final_flow * (model.final_concentration[0]+model.final_concentration[1])) / (pm['F_feed'] * (pm['c_feed_dict'][0]+pm['c_feed_dict'][1])) == model.recovery

    def water_recovery_rule(model):
        return 1 - (model.final_flow/(pm['F_feed']+sum(model.F_dilution[i] for i in pm['stages']))) == model.water_recovery

    def separation_factor_rule(model):
        return model.separation_factor == ((model.final_concentration[0]+model.final_concentration[1]) / (model.final_concentration[2]+model.final_concentration[3])) / ((pm['c_feed_dict'][0]+pm['c_feed_dict'][1]) / (pm['c_feed_dict'][2]+pm['c_feed_dict'][3]))

    model.recovery_constr = Constraint(rule=recovery_rule)
    model.water_recovery_constr = Constraint(rule=water_recovery_rule)
    model.separation_factor_constr = Constraint(rule=separation_factor_rule)


    # Constraint on dilution
    def dilution_rule(model):
        return sum(model.F_dilution[i] for i in pm['stages']) <= pm['F_dil_feed']

    model.dilution_constr = Constraint(rule=dilution_rule)


    # Constraints on pressure exchanger and power
    def power_rule(model):
        return model.power == (1/pm['pump_eff']) * pm['F_feed'] * model.p_pump + sum(model.power_dilution[i] for i in pm['stages'])

    def dilution_power_rule(model,st):
        return model.power_dilution[st] == (1/pm['pump_eff']) * model.F_dilution[st] * model.stages[st].p0

    def molar_power_rule(model):
        return model.mol_power == model.power / model.recovery

    def pex_rule(model):
        return model.p_pex == (pm['pex_eff'] * model.final_flow * model.final_pressure + pm['F_feed'] * model.p_pump)/pm['F_feed']

    model.power_constr = Constraint(rule=power_rule)
    model.dilution_power_constr = Constraint(pm['stages'],rule=dilution_power_rule)
    model.molar_power_rule = Constraint(rule=molar_power_rule)
    model.pex_constraint = Constraint(rule=pex_rule)


    # Multi-objective constraints
    if 'recovery' in constraints:
        def mo_rec(model):
            return model.recovery >= constraints['recovery']

        model.mo_rec = Constraint(rule=mo_rec)

    if 'separation_factor' in constraints:
        def mo_sf_rule(model):
            return model.separation_factor >= constraints['separation_factor']

        model.mo_sf = Constraint(rule=mo_sf_rule)
    
    if 'molar_power' in constraints:
        def mo_sp_rule(model):
            return model.mol_power <= constraints['molar_power']

        model.mo_sp = Constraint(rule=mo_sp_rule)


    # Objective
    if objective =='separation_factor':
        model.obj = Objective(expr=((model.final_concentration[2]+model.final_concentration[3]) / (sum(model.final_concentration[i] for i in range(4)))))
    elif objective == 'recovery':
        model.obj = Objective(expr=(1-model.recovery))
    elif objective =='molar_power':
        model.obj = Objective(expr=(model.mol_power))

    return model
    


def model_sdec_xrd(constraints, objective, pm, config='2d2s'):
    model = ConcreteModel()
    
    initialize_model(pm)
    # Problem-level variables

    model.recovery = Var(within=NonNegativeReals, bounds=(0,1), initialize = 0.0)
    model.separation_factor = Var(within=NonNegativeReals, bounds=(1,100), initialize = 1.0)
    model.water_recovery = Var(within=NonNegativeReals, bounds=(0,1), initialize = 0.0)
    model.p_pump = Var(within=NonNegativeReals, bounds = (0,pm['dp_max']), initialize = 0.0) # pressure generated by pump
    model.p_pex = Var(within=NonNegativeReals, bounds = (0,pm['dp_max']), initialize = 0.0) # pressure generated by pressure exchanger after the pump
    model.power = Var(within=NonNegativeReals, bounds = (0,2000), initialize = 1500.0) # power necessary
    model.mol_power = Var(within=NonNegativeReals, bounds = (0,3000), initialize = 1500.0)
    model.F_dilution = Var(pm['stages'],within=NonNegativeReals, bounds=(0,pm['F_dil_feed']), initialize = 0.0)
    model.omega = Var(pm['stages'],within=NonNegativeReals, bounds=(0,1), initialize = 1)
    model.recirc_power = Var(pm['stages'],within=NonNegativeReals, bounds=(0,2000), initialize = 0.0)
    model.power_dilution = Var(pm['stages'],within=NonNegativeReals, bounds = (0,2000), initialize = 0.0)

    model.final_flow = Var(within=NonNegativeReals, bounds=(0,pm['F_lim']), initialize = pm['F_feed'])
    model.final_pressure = Var(within=NonNegativeReals, bounds = (0,pm['dp_max']), initialize = 0.0)
    model.final_concentration = Var(pm['solutes'],within=NonNegativeReals, bounds = lambda b, i: pm['bounds']['C'][i], initialize = pm['c_feed_dict'])


    # Create pm['stages'] as Pyomo Blocks
    model.stages = Block(pm['stages'], rule=lambda b: stage_rule(b,pm))


    # Define mass balance and pressure constraints for each membrane element in each stage
    if config == '2d2s':
        def omb_0_rule(model):
            return model.stages[0].F0 == 0.5 * pm['F_feed'] + model.F_dilution[0] + model.omega[1] * model.stages[1].Fp
        def pre_0_rule(model):
            return model.stages[0].p0 == model.p_pex
        def cmb_0_rule(model,s):
            return model.stages[0].F0 * model.stages[0].C0[s] == 0.5 * pm['F_feed'] * pm['c_feed_dict'][s] + model.omega[1] * model.stages[1].Fp * model.stages[1].Cp[s]

        def omb_1_rule(model):
            return model.stages[1].F0 == model.stages[0].Fr + model.F_dilution[1] + 0.5 * model.omega[4] * model.stages[4].Fp
        def pre_1_rule(model):
            return model.stages[1].p0 == model.stages[0].pr
        def cmb_1_rule(model,s):
            return model.stages[1].F0 * model.stages[1].C0[s] == model.stages[0].Fr * model.stages[0].Cr[s] + 0.5 * model.omega[4] * model.stages[4].Fp * model.stages[4].Cp[s]
        def recirc_1_rule(model):
            return model.recirc_power[1] == (1/pm['pump_eff']) * model.stages[1].Fp * model.omega[1] * (model.stages[0].p0 - model.stages[1].pp)

        def omb_2_rule(model):
            return model.stages[2].F0 == 0.5 * pm['F_feed'] + model.F_dilution[2] + model.omega[3] * model.stages[3].Fp
        def pre_2_rule(model):
            return model.stages[2].p0 == model.p_pex
        def cmb_2_rule(model,s):
            return model.stages[2].F0 * model.stages[2].C0[s] == 0.5 * pm['F_feed'] * pm['c_feed_dict'][s] + model.omega[3] * model.stages[3].Fp * model.stages[3].Cp[s]

        def omb_3_rule(model):
            return model.stages[3].F0 == model.stages[2].Fr + model.F_dilution[3] + 0.5 * model.omega[4] * model.stages[4].Fp
        def pre_3_rule(model):
            return model.stages[3].p0 == model.stages[2].pr
        def cmb_3_rule(model,s):
            return model.stages[3].F0 * model.stages[3].C0[s] == model.stages[2].Fr * model.stages[2].Cr[s] + 0.5 * model.omega[4] * model.stages[4].Fp * model.stages[4].Cp[s]
        def recirc_3_rule(model):
            return model.recirc_power[3] == (1/pm['pump_eff']) * model.stages[3].Fp * model.omega[3] * (model.stages[2].p0 - model.stages[3].pp)

        def omb_4_rule(model):
            return model.stages[4].F0 == model.stages[1].Fr + model.stages[3].Fr + model.F_dilution[4] + model.omega[5] * model.stages[5].Fp
        def pre_4_rule(model):
            return model.stages[4].p0 == model.stages[1].pr
        def cmb_4_rule(model,s):
            return model.stages[4].F0 * model.stages[4].C0[s] == model.stages[1].Fr * model.stages[1].Cr[s] + model.stages[3].Fr * model.stages[3].Cr[s] + model.omega[5] * model.stages[5].Fp * model.stages[5].Cp[s]
        def recirc_4_rule(model):
            return model.recirc_power[4] == 0.5 * (1/pm['pump_eff']) * model.stages[4].Fp * model.omega[4] * (model.stages[1].p0 - model.stages[4].pp) + 0.5 * (1/pm['pump_eff']) * model.stages[4].Fp * model.omega[4] * (model.stages[3].p0 - model.stages[4].pp)

        def omb_5_rule(model):
            return model.stages[5].F0 == model.stages[4].Fr + model.F_dilution[5]
        def pre_5_rule(model):
            return model.stages[5].p0 == model.stages[4].pr
        def cmb_5_rule(model,s):
            return model.stages[5].F0 * model.stages[5].C0[s] == model.stages[4].Fr * model.stages[4].Cr[s]
        def recirc_5_rule(model):
            return model.recirc_power[5] == (1/pm['pump_eff']) * model.stages[5].Fp * model.omega[5] * (model.stages[4].p0 - model.stages[5].pp)

        def final_mix_rule(model):
            return model.final_flow == model.stages[5].Fr
        def final_pressure_rule(model):
            return model.final_pressure == model.stages[5].pr
        def final_component_rule(model,s):
            return model.final_flow * model.final_concentration[s] == model.stages[5].Fr * model.stages[5].Cr[s]
    else:
        print('Unknown config')
        return
    
    # Assignment
    model.omb_0 = Constraint(rule=omb_0_rule)
    model.cmb_0 = Constraint(pm['solutes'],rule=cmb_0_rule)
    model.omb_1 = Constraint(rule=omb_1_rule)
    model.cmb_1 = Constraint(pm['solutes'],rule=cmb_1_rule)
    model.omb_2 = Constraint(rule=omb_2_rule)
    model.cmb_2 = Constraint(pm['solutes'],rule=cmb_2_rule)
    model.omb_3 = Constraint(rule=omb_3_rule)
    model.cmb_3 = Constraint(pm['solutes'],rule=cmb_3_rule)
    model.omb_4 = Constraint(rule=omb_4_rule)
    model.cmb_4 = Constraint(pm['solutes'],rule=cmb_4_rule)
    model.omb_5 = Constraint(rule=omb_5_rule)
    model.cmb_5 = Constraint(pm['solutes'],rule=cmb_5_rule)

    model.pre_0 = Constraint(rule=pre_0_rule)
    model.pre_1 = Constraint(rule=pre_1_rule)
    model.pre_2 = Constraint(rule=pre_2_rule)
    model.pre_3 = Constraint(rule=pre_3_rule)
    model.pre_4 = Constraint(rule=pre_4_rule)
    model.pre_5 = Constraint(rule=pre_5_rule)

    model.recirc_const1 = Constraint(rule=recirc_1_rule)
    model.recirc_const3 = Constraint(rule=recirc_3_rule)
    model.recirc_const4 = Constraint(rule=recirc_4_rule)
    model.recirc_const5 = Constraint(rule=recirc_5_rule)


    model.final_mix_constr = Constraint(rule=final_mix_rule)
    model.final_pressure_constr = Constraint(rule=final_pressure_rule)
    model.final_component_constr = Constraint(pm['solutes'],rule=final_component_rule)


    # Performance metrics
    def recovery_rule(model):
        return (model.final_flow * (model.final_concentration[0]+model.final_concentration[1])) / (pm['F_feed'] * (pm['c_feed_dict'][0]+pm['c_feed_dict'][1])) == model.recovery

    def water_recovery_rule(model):
        return 1 - (model.final_flow/(pm['F_feed']+sum(model.F_dilution[i] for i in pm['stages']))) == model.water_recovery

    def separation_factor_rule(model):
        return model.separation_factor == ((model.final_concentration[0]+model.final_concentration[1]) / (model.final_concentration[2]+model.final_concentration[3])) / ((pm['c_feed_dict'][0]+pm['c_feed_dict'][1]) / (pm['c_feed_dict'][2]+pm['c_feed_dict'][3]))

    model.recovery_constr = Constraint(rule=recovery_rule)
    model.water_recovery_constr = Constraint(rule=water_recovery_rule)
    model.separation_factor_constr = Constraint(rule=separation_factor_rule)


    # Constraint on dilution
    def dilution_rule(model):
        return sum(model.F_dilution[i] for i in pm['stages']) <= pm['F_dil_feed']

    model.dilution_constr = Constraint(rule=dilution_rule)


    # Constraints on pressure exchanger and power
    def power_rule(model):
        return model.power == (1/pm['pump_eff']) * pm['F_feed'] * model.p_pump + model.recirc_power[1] +  model.recirc_power[3] +  model.recirc_power[4] +  model.recirc_power[5] + sum(model.power_dilution[i] for i in pm['stages'])

    def dilution_power_rule(model,st):
        return model.power_dilution[st] == (1/pm['pump_eff']) * model.F_dilution[st] * model.stages[st].p0

    def molar_power_rule(model):
        return model.mol_power == model.power / model.recovery

    def pex_rule(model):
        return model.p_pex == (pm['pex_eff'] * model.final_flow * model.final_pressure + pm['F_feed'] * model.p_pump)/pm['F_feed']

    model.power_constr = Constraint(rule=power_rule)
    model.dilution_power_constr = Constraint(pm['stages'],rule=dilution_power_rule)
    model.molar_power_rule = Constraint(rule=molar_power_rule)
    model.pex_constraint = Constraint(rule=pex_rule)


    # Multi-objective constraints
    if 'recovery' in constraints:
        def mo_rec(model):
            return model.recovery >= constraints['recovery']

        model.mo_rec = Constraint(rule=mo_rec)

    if 'separation_factor' in constraints:
        def mo_sf_rule(model):
            return model.separation_factor >= constraints['separation_factor']

        model.mo_sf = Constraint(rule=mo_sf_rule)
    
    if 'molar_power' in constraints:
        def mo_sp_rule(model):
            return model.mol_power <= constraints['molar_power']

        model.mo_sp = Constraint(rule=mo_sp_rule)


    # Objective
    if objective =='separation_factor':
        model.obj = Objective(expr=((model.final_concentration[2]+model.final_concentration[3]) / (sum(model.final_concentration[i] for i in range(4)))))
    elif objective == 'recovery':
        model.obj = Objective(expr=(1-model.recovery))
    elif objective =='molar_power':
        model.obj = Objective(expr=(model.mol_power))

    return model
    

## OPTIMIZATION AND EXTRACTION #################################################################################################################

def opti(model,solver = 'ipopt'):
    # # Solver call - IPOPT
    try:
        if solver == 'ipopt':
            with SolverFactory('ipopt') as opt:
                results = opt.solve(model, tee=True)
        elif solver == 'baron':
            with SolverFactory('baron') as opt:
                results = opt.solve(model, tee=True, options={"MaxTime": -1})

        if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
            optimal = 1
        else:
            optimal = 0
    except ValueError:
        results = {}
        optimal = 0

    return model, results, optimal


def extract_results(model,n_stages):
    results = {}
    try:
        results['p_feed'] = value(model.p_feed)
    except:
        pass
    try:
        results['p_pump'] = value(model.p_pump)
        results['p_pex'] = value(model.p_pex)
    except:
        pass

    p_perm = {}
    for i in range(n_stages):
        p_perm[i] = value(model.stages[i].pp)
    results['p_perm'] = p_perm

    try:
        omegas = {}
        for i in range(n_stages):
            omegas[i] = value(model.omega[i])
        results['rec_split'] = omegas
    except:
        pass

    try:
        dilutions = {}
        for i in range(n_stages):
            dilutions[i] = value(model.F_dilution[i])
        results['dilution'] = dilutions
    except:
        pass

    results['sep_factor'] = value(model.separation_factor)
    results['mol_power'] = value(model.mol_power)
    results['recovery'] = value(model.recovery)
    results['water_recovery'] = value(model.water_recovery)

    return results


def log_opti(model):
    import logging

    # Get the logger
    logger = logging.getLogger()

    # Set the logging level to include INFO messages
    logger.setLevel(logging.INFO)
    log_infeasible_constraints(model)

    pyomo.util.infeasible.log_close_to_bounds(model)