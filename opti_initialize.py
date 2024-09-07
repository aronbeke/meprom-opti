import sde_model
import nf_simulation
import numpy as np
import pandas as pd
import warnings


def load_swcc_problem(dp_max,no_collocation,no_nodes):
    ### SYSTEM PARAMETERS
    pm = {
        'F_feed': 20.0,  # Volumetric flow rate of the feed stream m3/h
        'F_dil_feed': 30.0,
        'A': 40.0,  # m2
        'l_module': 0.965,  # m
        'eta': 3.132,  # kg/mh
        'l_mesh': 0.00327,  # m
        'df': 0.0003395,  # m (fitted from pressure drops)
        'theta': 1.83259571,  # rad
        'n_env': 28,
        'b_env': 0.85,  # m
        'h': 0.0004008,  # m (fitted from pressure drops)
        'rho': 1030,  # kg/m3 (assumed)
        'T': 303.0,  # K
        'dp_max': dp_max,  # bar
        'pump_eff': 0.85,  # pump efficiency
        'pex_eff': 0.85,  # pressure exchanger efficiency
        'P1': -1.079e-4,  # m/hbar2
        'P0': 5.891e-3,  # m/hbar
        'OB2': 4.442e-8,
        'OB1': -1.484e-4,
        'OB0': 1.051,
        'OP1': 1.608e-5,
        'OP0': 0.9215,
        'b0': 0.20,  # concentration polarization coeff.
        'sf': False,  # solution-friction
    }

    # Concentrations in the feed stream #mol / m3

    pm['F_lim'] = pm['F_feed'] + pm['F_dil_feed']

    pm['c_feed'] = [10.925,60,600.173913,10.46153846,31.38541667,685.6901408]
    pm['c_feed_dict'] = {0: 10.925, 1: 60, 2: 600.173913, 3: 10.46153846, 4: 31.38541667, 5: 685.6901408}
    pm['Li_list'] = [2.58224237e-03, 5.42316649e-04, 2.87688295e-03, 4.14596285e-02, 1.60533575e-05, 2.07063310e-01]
    pm['Ki_list'] = [2.17309816e-03,6.80802898e-03, 9.77814275e-01 ,5.51734781e-01, 9.36038188e-01 ,2.16476691e-01]
    pm['zi_list'] = [2,2,1,1,-2,-1]

    # Collocation and discretization parameters 
    pm['ns'] = len(pm['c_feed']) # no. of solutes
    pm['nk'] = no_collocation # no. of collocation points
    pm['ne'] = 3 # no. of elements per stage
    pm['nst'] = 6 # no. of stages in configuration
    pm['nn'] = no_nodes # no. of nodes per element

    pm_sim = pm.copy()
    pm_sim['P1'] = pm_sim['P1'] / 1e10
    pm_sim['P0'] = pm_sim['P0'] / 1e5
    pm_sim['dp_max'] = pm_sim['dp_max'] * 1e5

    pm1 = pm.copy()
    pm1_sim = pm_sim.copy()
    pm1['nn'] = 1
    pm1_sim['nn'] = 1

    return pm, pm_sim


def generate_random_combinations(total_sum, num_max, sample_size):
    '''
    Generate random process parameters
    '''
    num_set = np.random.uniform(0,num_max,size=(sample_size))
    while np.sum(num_set) >= total_sum:
        num_set = np.random.uniform(0,num_max,size=(sample_size))
    return list(num_set)

def random_initialization(model,pm_sim,init_pm,model_x=False,model_d=False,model_r = False):
    '''
    Main function to be called. Loops with random process parameters until finds a feasible one.
    '''
    feasible = 0
    warnings.simplefilter('error', RuntimeWarning)
    while feasible == 0:
        try:
            p_feed = np.random.uniform(init_pm['p_feed_min'],init_pm['p_feed_max'])*1e5 # Pa
            pp_list = list(np.random.uniform(0,p_feed/1e5,size=(pm_sim['nst']))*1e5) # Pa
            if model_d:
                F_dil_list = generate_random_combinations(pm_sim['F_dil_feed'],init_pm['dilution_max'],pm_sim['nst']) #m3/h
            else:
                F_dil_list = [0] * pm_sim['nst']
            if model_r:
                split_list = list(np.random.uniform(0,1,size=(pm_sim['nst'])))
            else:
                split_list = [0] * pm_sim['nst']
            init_parameters = [p_feed, pp_list, F_dil_list, split_list]
            print('Initializing with ',init_parameters)
            model, status = sdec_2d2s_initialization(model,init_parameters,pm_sim,model_x=model_x,model_d=model_d,model_r = model_r)
        except RuntimeWarning:
             status = True
        
        if status:
            feasible = 0
        else:
            feasible = 1
    
    return model, init_parameters


def reject_initialization(desc,sol_dict,n_stages,n_solutes,n_elements):
    '''
    Checks whether simulation based on random process parameters is feasible
    '''
    init_error = False
    recovery,water_recovery,sep_factor,p_pump,power,mol_power,dil_power,rec_power = desc
    for i in range(n_stages):
        if dil_power[i] < 0:
            init_error = True
            break
        if rec_power[i] < 0:
            init_error = True
            break

    for el in [recovery,water_recovery,sep_factor,p_pump,power,mol_power]:
        if el < 0:
            init_error = True
            break

    for i in range(n_stages):
        for d in ['Fr','Fp','F0','p0','pr','pp']:
            if sol_dict[i][d] < 0:
                init_error = True
                break
        for d in ['c0','cr','cp']:
            for j in range(n_solutes):
                if sol_dict[i][d][j] < 0:
                    init_error = True
                    break
        
        for el in range(n_elements):
            if sol_dict[i]['elements'][el]['p0'] - sol_dict[i]['elements'][el]['pressure_drop'] - sol_dict[i]['elements'][el]['pp'] < 0:
                init_error = True
                break
            for d in ['Fr','Fp','F0','p0','pr','pp']:
                if sol_dict[i]['elements'][el][d] < 0:
                    # print('Infeasible element: ',el,sol_dict[i]['elements'][el][d])
                    init_error = True
                    break
            for d in ['c0','cr','cp']:
                for j in range(n_solutes):
                    if sol_dict[i]['elements'][el][d][j] < 0:
                        # print('Infeasible element: ',el,sol_dict[i]['elements'][el][d])
                        init_error = True
                        break

    return init_error


def sdec_2d2s_initialization(model,init_parameters,pm_sim,model_x=False,model_d=False,model_r = False):
    '''
    Initializes model. Rejects initialization if infeasible.
    Status: False = ok, True = infeasible initialization
    model_type: 'xrd', 'xd', 'x', '0'
    x: pressure exchange
    r: recycle (not considered in initialization)
    d: dilution
    0: conventional linear cascade
    '''

    status = False
    n_stages = pm_sim['nst']
    n_solutes = pm_sim['ns']
    n_elements = pm_sim['ne']

    p_feed, pp_list, F_dil_list, split_list = init_parameters
    if model_r:
        act_init_pm = init_parameters
        desc, sol_dict = nf_simulation.sdec_r_2d2s_simulation(act_init_pm, pm_sim,model_x=model_x,model_d=model_d)
        print('Simulation done')
    else:
        act_init_pm = [p_feed, pp_list, F_dil_list]
        desc, sol_dict = nf_simulation.sdec_2d2s_simulation(act_init_pm, pm_sim,model_x=model_x,model_d=model_d)
        print('Simulation done')
    
    recovery,water_recovery,sep_factor,p_pump,power,mol_power,dil_power,rec_power = desc

    status = reject_initialization(desc,sol_dict,n_stages,n_solutes,n_elements)
    if status:
        return model, status
    print("Feasible model found")

    model.recovery.value = recovery
    model.separation_factor.value = sep_factor
    model.water_recovery.value = water_recovery
    if model_x == False:
        model.p_feed = p_feed /1e5
    else:
        model.p_pump.value = p_pump /1e5
        model.p_pex.value = p_feed /1e5
    model.power.value = power /1e5
    model.mol_power.value = mol_power /1e5

    if model_d:
        for i in range(n_stages):
            model.F_dilution[i].value = F_dil_list[i]
            model.power_dilution[i].value = dil_power[i] /1e5
    if model_r:
        for i in range(n_stages):
            model.omega[i].value = split_list[i]
            model.recirc_power[i].value = rec_power[i] /1e5

    model.final_flow.value = sol_dict[5]['Fr']
    model.final_pressure.value = sol_dict[5]['pr'] /1e5
    for i in range(n_solutes):
        model.final_concentration[i].value = sol_dict[5]['cr'][i]

    for st in range(n_stages):
        model.stages[st].p0.value = sol_dict[st]['p0'] /1e5
        model.stages[st].pp.value = sol_dict[st]['pp'] /1e5
        model.stages[st].pr.value = sol_dict[st]['pr'] /1e5
        model.stages[st].F0.value = sol_dict[st]['F0']
        model.stages[st].Fr.value = sol_dict[st]['Fr']
        model.stages[st].Fp.value = sol_dict[st]['Fp']
        for i in range(n_solutes):
            model.stages[st].C0[i].value = sol_dict[st]['c0'][i]
            model.stages[st].Cr[i].value = sol_dict[st]['cr'][i]
            model.stages[st].Cp[i].value = sol_dict[st]['cp'][i]

        for el in range(pm_sim['ne']):
            model.stages[st].elems[el].p0.value = sol_dict[st]['elements'][el]['p0'] /1e5
            model.stages[st].elems[el].pp.value = sol_dict[st]['elements'][el]['pp'] /1e5
            model.stages[st].elems[el].pdrop.value = (sol_dict[st]['elements'][el]['p0'] - sol_dict[st]['elements'][el]['pr']) /1e5
            model.stages[st].elems[el].beta.value = sol_dict[st]['elements'][el]['beta']
            model.stages[st].elems[el].F0.value = sol_dict[st]['elements'][el]['F0']
            model.stages[st].elems[el].Fr.value = sol_dict[st]['elements'][el]['Fr']
            model.stages[st].elems[el].Fp.value = sol_dict[st]['elements'][el]['Fp']
            dp = (sol_dict[st]['elements'][el]['pr'] - sol_dict[st]['elements'][el]['pp']) /1e5
            for i in range(n_solutes):
                model.stages[st].elems[el].C0[i].value = sol_dict[st]['elements'][el]['c0'][i]
                model.stages[st].elems[el].Cr[i].value = sol_dict[st]['elements'][el]['cr'][i]
                model.stages[st].elems[el].Cp[i].value = sol_dict[st]['elements'][el]['cp'][i]

            for no in range(pm_sim['nn']):
                j_list = sol_dict[st]['elements'][el]['nodes'][no]['sim'][3:(3+n_solutes)]
                cr_list = sol_dict[st]['elements'][el]['nodes'][no]['sim'][(3+n_solutes):(3+2*n_solutes)]
                cm_list = sol_dict[st]['elements'][el]['nodes'][no]['sim'][(3+2*n_solutes):(3+3*n_solutes)]
                theta_phi_reduced_list = sol_dict[st]['elements'][el]['nodes'][no]['sim'][(3+3*n_solutes):(3+3*n_solutes+(pm_sim['nk']-1))]
                c_reduced_list = sol_dict[st]['elements'][el]['nodes'][no]['sim'][(3+3*n_solutes+(pm_sim['nk']-1)):(3+3*n_solutes+(pm_sim['nk']-1)+(pm_sim['nk']-1)*n_solutes)]

                j, cr, cm = np.array([j_list]), np.transpose(np.array([cr_list])), np.transpose(np.array([cm_list]))
                theta_phi_reduced_vector, c_reduced_vector = np.transpose(np.array([theta_phi_reduced_list])), np.array([c_reduced_list])
                c0_vector = np.transpose(np.array([ sol_dict[st]['elements'][el]['nodes'][no]['c0']]))
                theta_phi_vector = np.vstack((np.array([[0]]),theta_phi_reduced_vector))

                C_reduced_shape = (pm_sim['nk']-1,n_solutes)

                J = np.tile(j,(pm_sim['nk']-1,1))
                V = np.vander(np.linspace(0,1,pm_sim['nk']), pm_sim['nk'], increasing=True)
                V_inv = np.linalg.inv(V)
                D_coeff = np.tile(np.array([range(pm_sim['nk'])]),(pm_sim['nk']-1,1))
                D_vander = np.hstack((np.zeros((pm_sim['nk']-1,1)),np.vander(np.linspace(0,1,pm_sim['nk']-1), pm_sim['nk']-1, increasing=True)))
                D = np.multiply(D_coeff,D_vander)
                C_reduced = c_reduced_vector.reshape(C_reduced_shape)
                C = np.vstack((np.transpose(cm),C_reduced))
                Phi = np.tile(theta_phi_vector,(1,n_solutes))

                DPhiDx = D @ (V_inv @ Phi)
                DCDx = D @ (V_inv @ C)

                model.stages[st].elems[el].nodes[no].P.value =  (pm_sim['P1']*(dp*1e5) + pm_sim['P0']) / 1e5
                model.stages[st].elems[el].nodes[no].omega_brine.value = sde_model.osmotic_coefficient_brine_tds(cm,pm_sim['OB2'],pm_sim['OB1'],pm_sim['OB0'])
                model.stages[st].elems[el].nodes[no].omega_perm.value = sde_model.osmotic_coefficient_perm_tds(C[-1,:],pm_sim['OP1'],pm_sim['OP0'])
                Fr_node = sol_dict[st]['elements'][el]['nodes'][no]['sim'][1]
                Fp_node = sol_dict[st]['elements'][el]['nodes'][no]['sim'][2]
                F0_node = sol_dict[st]['elements'][el]['nodes'][no]['F0']
                model.stages[st].elems[el].nodes[no].beta.value = np.exp(pm_sim['b0']*(Fp_node/F0_node))
                model.stages[st].elems[el].nodes[no].dp.value = dp
                model.stages[st].elems[el].nodes[no].Fr.value = Fr_node
                model.stages[st].elems[el].nodes[no].Fp.value = Fp_node
                model.stages[st].elems[el].nodes[no].F0.value = F0_node
                model.stages[st].elems[el].nodes[no].Jv.value = sol_dict[st]['elements'][el]['nodes'][no]['sim'][0]
                for j in range(pm_sim['nk']):
                    model.stages[st].elems[el].nodes[no].Phi[j].value = Phi[j,0]
                for j in range(1,pm_sim['nk']):
                    model.stages[st].elems[el].nodes[no].dPhidx[j].value = DPhiDx[j-1,0]
                for i in range(n_solutes):
                    model.stages[st].elems[el].nodes[no].J[i].value = j_list[i]
                    model.stages[st].elems[el].nodes[no].C0[i].value = sol_dict[st]['elements'][el]['nodes'][no]['c0'][i]
                    model.stages[st].elems[el].nodes[no].Cr[i].value = cr_list[i]
                    for j in range(pm_sim['nk']):
                        model.stages[st].elems[el].nodes[no].C[j,i].value = C[j,i]
                    for j in range(1,pm_sim['nk']):
                        model.stages[st].elems[el].nodes[no].dCdx[j,i].value = DCDx[j-1,i]

    return model, status


def variable_value_transfer(origin_model,target_model,n_solutes,n_coll,n_nodes_target,n_elements,n_stages,model_x=False,model_d=False,model_r = False,same_nodes=False):
    '''
    Transfer pyomo model variables between models of different complexity (discretization)
    '''
    # origin has 1 node per element
    target_model.recovery.value = origin_model.recovery.value
    target_model.separation_factor.value = origin_model.separation_factor.value
    target_model.water_recovery.value = origin_model.water_recovery.value
    if model_x == False:
        target_model.p_feed.value = origin_model.p_feed.value
    else:
        target_model.p_pump.value = origin_model.p_pump.value
        target_model.p_pex.value = origin_model.p_pex.value
    target_model.power.value = origin_model.power.value
    target_model.mol_power.value = origin_model.mol_power.value
    for i in range(n_stages):
        if model_d:
            target_model.F_dilution[i].value = origin_model.F_dilution[i].value
            target_model.power_dilution[i].value = origin_model.power_dilution[i].value
        if model_r:
            target_model.omega[i].value = origin_model.omega[i].value
            target_model.recirc_power[i].value = origin_model.recirc_power[i].value

    target_model.final_flow.value = origin_model.final_flow.value
    target_model.final_pressure.value = origin_model.final_pressure.value
    for i in range(n_solutes):
        target_model.final_concentration[i].value = origin_model.final_concentration[i].value

    for st in range(n_stages):
        target_model.stages[st].p0.value = origin_model.stages[st].p0.value
        target_model.stages[st].pp.value = origin_model.stages[st].p0.value
        target_model.stages[st].pr.value = origin_model.stages[st].p0.value
        target_model.stages[st].F0.value = origin_model.stages[st].p0.value
        target_model.stages[st].Fr.value = origin_model.stages[st].p0.value
        target_model.stages[st].Fp.value = origin_model.stages[st].p0.value
        for i in range(n_solutes):
            target_model.stages[st].C0[i].value = origin_model.stages[st].C0[i].value
            target_model.stages[st].Cr[i].value = origin_model.stages[st].Cr[i].value
            target_model.stages[st].Cp[i].value = origin_model.stages[st].Cp[i].value

        for el in range(n_elements):
            target_model.stages[st].elems[el].p0.value = origin_model.stages[st].elems[el].p0.value
            target_model.stages[st].elems[el].pp.value = origin_model.stages[st].elems[el].p0.value
            target_model.stages[st].elems[el].pdrop.value = origin_model.stages[st].elems[el].pdrop.value
            target_model.stages[st].elems[el].beta.value = origin_model.stages[st].elems[el].pdrop.value
            target_model.stages[st].elems[el].F0.value = origin_model.stages[st].elems[el].p0.value
            target_model.stages[st].elems[el].Fr.value = origin_model.stages[st].elems[el].p0.value
            target_model.stages[st].elems[el].Fp.value = origin_model.stages[st].elems[el].p0.value
            for i in range(n_solutes):
                target_model.stages[st].elems[el].C0[i].value = origin_model.stages[st].elems[el].C0[i].value
                target_model.stages[st].elems[el].Cr[i].value = origin_model.stages[st].elems[el].Cr[i].value
                target_model.stages[st].elems[el].Cp[i].value = origin_model.stages[st].elems[el].Cp[i].value

            if not same_nodes:
                for no in range(n_nodes_target):
                    target_model.stages[st].elems[el].nodes[no].P.value = origin_model.stages[st].elems[el].nodes[0].P.value
                    target_model.stages[st].elems[el].nodes[no].omega_brine.value = origin_model.stages[st].elems[el].nodes[0].omega_brine.value
                    target_model.stages[st].elems[el].nodes[no].omega_perm.value = origin_model.stages[st].elems[el].nodes[0].omega_perm.value
                    target_model.stages[st].elems[el].nodes[no].beta.value = origin_model.stages[st].elems[el].nodes[0].beta.value
                    target_model.stages[st].elems[el].nodes[no].dp.value = origin_model.stages[st].elems[el].nodes[0].dp.value
                    target_model.stages[st].elems[el].nodes[no].Fr.value = origin_model.stages[st].elems[el].nodes[0].Fr.value
                    target_model.stages[st].elems[el].nodes[no].Fp.value = origin_model.stages[st].elems[el].nodes[0].Fp.value
                    target_model.stages[st].elems[el].nodes[no].F0.value = origin_model.stages[st].elems[el].nodes[0].F0.value
                    target_model.stages[st].elems[el].nodes[no].Jv.value = origin_model.stages[st].elems[el].nodes[0].Jv.value
                    for j in range(n_coll):
                        target_model.stages[st].elems[el].nodes[no].Phi[j].value = origin_model.stages[st].elems[el].nodes[0].Phi[j].value
                    for j in range(1,n_coll):
                        target_model.stages[st].elems[el].nodes[no].dPhidx[j].value = origin_model.stages[st].elems[el].nodes[0].dPhidx[j].value
                    for i in range(n_solutes):
                        target_model.stages[st].elems[el].nodes[no].J[i].value = origin_model.stages[st].elems[el].nodes[0].J[i].value
                        target_model.stages[st].elems[el].nodes[no].C0[i].value = origin_model.stages[st].elems[el].nodes[0].C0[i].value
                        target_model.stages[st].elems[el].nodes[no].Cr[i].value = origin_model.stages[st].elems[el].nodes[0].Cr[i].value
                        for j in range(n_coll):
                            target_model.stages[st].elems[el].nodes[no].C[j,i].value = origin_model.stages[st].elems[el].nodes[0].C[j,i].value
                        for j in range(1,n_coll):
                            target_model.stages[st].elems[el].nodes[no].dCdx[j,i].value = origin_model.stages[st].elems[el].nodes[0].dCdx[j,i].value
            else:
                for no in range(n_nodes_target):
                    target_model.stages[st].elems[el].nodes[no].P.value = origin_model.stages[st].elems[el].nodes[no].P.value
                    target_model.stages[st].elems[el].nodes[no].omega_brine.value = origin_model.stages[st].elems[el].nodes[no].omega_brine.value
                    target_model.stages[st].elems[el].nodes[no].omega_perm.value = origin_model.stages[st].elems[el].nodes[no].omega_perm.value
                    target_model.stages[st].elems[el].nodes[no].beta.value = origin_model.stages[st].elems[el].nodes[no].beta.value
                    target_model.stages[st].elems[el].nodes[no].dp.value = origin_model.stages[st].elems[el].nodes[no].dp.value
                    target_model.stages[st].elems[el].nodes[no].Fr.value = origin_model.stages[st].elems[el].nodes[no].Fr.value
                    target_model.stages[st].elems[el].nodes[no].Fp.value = origin_model.stages[st].elems[el].nodes[no].Fp.value
                    target_model.stages[st].elems[el].nodes[no].F0.value = origin_model.stages[st].elems[el].nodes[no].F0.value
                    target_model.stages[st].elems[el].nodes[no].Jv.value = origin_model.stages[st].elems[el].nodes[no].Jv.value
                    for j in range(n_coll):
                        target_model.stages[st].elems[el].nodes[no].Phi[j].value = origin_model.stages[st].elems[el].nodes[no].Phi[j].value
                    for j in range(1,n_coll):
                        target_model.stages[st].elems[el].nodes[no].dPhidx[j].value = origin_model.stages[st].elems[el].nodes[no].dPhidx[j].value
                    for i in range(n_solutes):
                        target_model.stages[st].elems[el].nodes[no].J[i].value = origin_model.stages[st].elems[el].nodes[no].J[i].value
                        target_model.stages[st].elems[el].nodes[no].C0[i].value = origin_model.stages[st].elems[el].nodes[no].C0[i].value
                        target_model.stages[st].elems[el].nodes[no].Cr[i].value = origin_model.stages[st].elems[el].nodes[no].Cr[i].value
                        for j in range(n_coll):
                            target_model.stages[st].elems[el].nodes[no].C[j,i].value = origin_model.stages[st].elems[el].nodes[no].C[j,i].value
                        for j in range(1,n_coll):
                            target_model.stages[st].elems[el].nodes[no].dCdx[j,i].value = origin_model.stages[st].elems[el].nodes[no].dCdx[j,i].value
            
    return origin_model,target_model


def sim_validation(input_file,dp_max,no_collocation,no_nodes,model_x=False,model_d=False,model_r=False):
    df = pd.read_csv(input_file+'.csv')

    length = len(df.index)
    SIM_WATER_REC = np.zeros(length)
    SIM_BIVALENT_REC = np.zeros(length)
    SIM_SEP_FACTOR = np.zeros(length)
    SIM_MOL_POWER = np.zeros(length)

    pm, pm_sim = load_swcc_problem(dp_max,no_collocation,no_nodes)

    for i in range(length):
        p_feed = df['FEED_PRESSURE'].iloc[i]*1e5
        pp_list = list(np.array([df['PERMEATE_PRESSURE_0'].iloc[i], df['PERMEATE_PRESSURE_1'].iloc[i], df['PERMEATE_PRESSURE_2'].iloc[i], df['PERMEATE_PRESSURE_3'].iloc[i], df['PERMEATE_PRESSURE_4'].iloc[i], df['PERMEATE_PRESSURE_5'].iloc[i]])*1e5)
        F_dil_list = [df['DILUTION_0'].iloc[i], df['DILUTION_1'].iloc[i], df['DILUTION_2'].iloc[i], df['DILUTION_3'].iloc[i], df['DILUTION_4'].iloc[i], df['DILUTION_5'].iloc[i]]
        split_list = [0, df['OMEGA_1'].iloc[i], 0, df['OMEGA_3'].iloc[i], df['OMEGA_4'].iloc[i], df['OMEGA_5'].iloc[i]]

        if model_r:
            init_parameters = [p_feed, pp_list, F_dil_list,split_list]
            desc, sol_dict = nf_simulation.sdec_r_2d2s_simulation(init_parameters,pm_sim,model_x=model_x,model_d=model_d)
        else:
            init_parameters = [p_feed, pp_list, F_dil_list]
            desc, sol_dict = nf_simulation.sdec_2d2s_simulation(init_parameters,pm_sim,model_x=model_x,model_d=model_d)

        recovery,water_recovery,sep_factor,p_pump,power,mol_power,dil_power, rec_power = desc

        SIM_WATER_REC[i] = water_recovery
        SIM_BIVALENT_REC[i] = recovery
        SIM_SEP_FACTOR[i] = sep_factor
        SIM_MOL_POWER[i] = mol_power

    df['SIM_WATER_REC'] = SIM_WATER_REC
    df['SIM_BIVALENT_REC'] = SIM_BIVALENT_REC
    df['SIM_SEP_FACTOR'] = SIM_SEP_FACTOR
    df['SIM_MOL_POWER'] = SIM_MOL_POWER

    df.to_csv(input_file+'_validation.csv')