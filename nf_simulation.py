import sde_model
import numpy as np
import pandas as pd
import scipy

'''
Nanofiltration cascade simulations.
Main focus here: SDEC, SF, SDE models
'''

def sdec_stage_sim(stage_parameters, constants):
    F_feed, c_feed, A, T, nk, nn, ne, p0, pp, F_dil = stage_parameters

    ns = len(c_feed)
    F0 = F_feed + F_dil
    c0 = []
    for c in c_feed:
        c0.append(F_feed*c/F0)

    solutions = []
    stage_solutions = {}
    
    parameters = [F0, c0, A, T, nk, ns, p0, pp]
    arguments = [parameters, constants]
    els = sde_model.sdec_spiral_wound_mesh_module(nn,arguments)
    solutions.append(els)

    for i in range(ne-1):
        parameters = [solutions[-1]['Fr'], solutions[-1]['cr'], A, T, nk, ns, solutions[-1]['pr'], pp]
        arguments = [parameters, constants]
        els = sde_model.sdec_spiral_wound_mesh_module(nn,arguments)
        solutions.append(els)

    Fp_final = np.sum(i['Fp'] for i in solutions)
    cp_final = []
    for i in range(len(c_feed)):
        cp_final.append(np.sum(j['Fp']*j['cp'][i] for j in solutions)/Fp_final)

    stage_solutions['Fr'] = solutions[-1]['Fr']
    stage_solutions['cr'] = solutions[-1]['cr']
    stage_solutions['pr'] = solutions[-1]['pr']
    stage_solutions['F0'] = F0
    stage_solutions['c0'] = c0
    stage_solutions['p0'] = p0
    stage_solutions['Fp'] = Fp_final
    stage_solutions['cp'] = cp_final
    stage_solutions['pp'] = pp
    stage_solutions['elements'] = solutions
    return stage_solutions

def sdec_r_stage_sim(stage_parameters, constants):
    F_feed, c_feed, A, T, nk, nn, ne, p0, pp, F_dil, F_rec, c_rec = stage_parameters

    ns = len(c_feed)
    F0 = F_feed + F_dil + F_rec
    c0 = []
    for i in range(len(c_feed)):
        c0.append((F_feed*c_feed[i] + F_rec*c_rec[i])/F0)

    solutions = []
    stage_solutions = {}
    
    parameters = [F0, c0, A, T, nk, ns, p0, pp]
    # print(parameters)
    arguments = [parameters, constants]
    els = sde_model.sdec_spiral_wound_mesh_module(nn,arguments)
    solutions.append(els)

    for i in range(ne-1):
        parameters = [solutions[-1]['Fr'], solutions[-1]['cr'], A, T, nk, ns, solutions[-1]['pr'], pp]
        # print(parameters)
        arguments = [parameters, constants]
        els = sde_model.sdec_spiral_wound_mesh_module(nn,arguments)
        solutions.append(els)

    Fp_final = np.sum(i['Fp'] for i in solutions)
    cp_final = []
    for i in range(len(c_feed)):
        cp_final.append(np.sum(j['Fp']*j['cp'][i] for j in solutions)/Fp_final)

    stage_solutions['Fr'] = solutions[-1]['Fr']
    stage_solutions['cr'] = solutions[-1]['cr']
    stage_solutions['pr'] = solutions[-1]['pr']
    stage_solutions['F0'] = F0
    stage_solutions['c0'] = c0
    stage_solutions['p0'] = p0
    stage_solutions['Fp'] = Fp_final
    stage_solutions['cp'] = cp_final
    stage_solutions['pp'] = pp
    stage_solutions['elements'] = solutions
    return stage_solutions

##########################

def sdec_2d2s_simulation(process_pm,pm,model_x=False,model_d=False):
    sol_dict = {}
    p_feed, pp_list, F_dil_list = process_pm

    if model_d == False:
        for i in range(len(F_dil_list)):
            F_dil_list[i] = 0
    
    constants = [pm['Li_list'], pm['zi_list'], pm['P1'], pm['P0'], pm['OB2'], pm['OB1'], pm['OB0'], pm['OP1'], pm['OP0'], pm['b0'], pm['h'],pm['rho'],pm['l_module'],pm['eta'],pm['l_mesh'],pm['df'],pm['theta'],pm['n_env'],pm['b_env'],pm['Ki_list'],pm['sf']]

    # 1st stages
    stage_parameters_0 = [0.5 * pm['F_feed'], pm['c_feed'], pm['A'], pm['T'], pm['nk'], pm['nn'], pm['ne'], p_feed, pp_list[0], F_dil_list[0]]
    sol_dict[0] = sdec_stage_sim(stage_parameters_0, constants)

    stage_parameters_2 = [0.5 * pm['F_feed'], pm['c_feed'], pm['A'], pm['T'], pm['nk'], pm['nn'], pm['ne'], p_feed, pp_list[2], F_dil_list[2]]
    sol_dict[2] = sdec_stage_sim(stage_parameters_2, constants)

    # 2nd stages
    stage_parameters_1 = [sol_dict[0]['Fr'], sol_dict[0]['cr'], pm['A'], pm['T'], pm['nk'], pm['nn'], pm['ne'], sol_dict[0]['pr'], pp_list[1], F_dil_list[1]]
    sol_dict[1] = sdec_stage_sim(stage_parameters_1, constants)

    stage_parameters_3 = [sol_dict[2]['Fr'], sol_dict[2]['cr'], pm['A'], pm['T'], pm['nk'], pm['nn'], pm['ne'], sol_dict[2]['pr'], pp_list[3], F_dil_list[3]]
    sol_dict[3] = sdec_stage_sim(stage_parameters_3, constants) 

    # 3rd stage
    F0_4 = sol_dict[1]['Fr'] + sol_dict[3]['Fr']
    c0_4 = []
    for i in range(len(pm['c_feed'])):
        c0_4.append((sol_dict[1]['Fr']*sol_dict[1]['cr'][i] + sol_dict[3]['Fr']*sol_dict[3]['cr'][i])/F0_4)
    
    stage_parameters_4 = [F0_4, c0_4, pm['A'], pm['T'], pm['nk'], pm['nn'], pm['ne'], sol_dict[1]['pr'], pp_list[4], F_dil_list[4]]
    sol_dict[4] = sdec_stage_sim(stage_parameters_4, constants)

    #4th stage
    stage_parameters_5 = [sol_dict[4]['Fr'], sol_dict[4]['cr'], pm['A'], pm['T'], pm['nk'], pm['nn'], pm['ne'], sol_dict[4]['pr'], pp_list[5], F_dil_list[5]]
    sol_dict[5] = sdec_stage_sim(stage_parameters_5, constants)

    # Performance descriptors
    recovery = (sol_dict[5]['Fr'] * (sol_dict[5]['cr'][0] + sol_dict[5]['cr'][1])) / (pm['F_feed'] * (pm['c_feed'][0]+ pm['c_feed'][1]))
    water_recovery = 1 - (sol_dict[5]['Fr']/(pm['F_feed'] + np.sum(F_dil_list)))
    sep_factor = ((sol_dict[5]['cr'][0] + sol_dict[5]['cr'][1]) / (sol_dict[5]['cr'][2] + sol_dict[5]['cr'][3])) / ((pm['c_feed'][0]+pm['c_feed'][1]) / (pm['c_feed'][2]+pm['c_feed'][3]))
    if model_x:
        p_pump = (pm['F_feed']*p_feed - pm['pex_eff']*sol_dict[5]['Fr']*sol_dict[5]['pr'])/pm['F_feed']
    else: 
        p_pump = p_feed

    dil_power = {}
    for i in range(len(F_dil_list)):
        dil_power[i] = (1/pm['pump_eff']) * F_dil_list[i] * sol_dict[i]['p0']
    
    power = (1/pm['pump_eff']) * pm['F_feed'] * p_pump + np.sum(dil_power[i] for i in range(len(F_dil_list)))
    mol_power = power / recovery

    rec_power = [0]*pm['nst']
    
    desc = [recovery,water_recovery,sep_factor,p_pump,power,mol_power,dil_power,rec_power]

    return desc, sol_dict


def sdec_r_2d2s_simulation(process_pm,pm,model_x=False,model_d=False):
    sol_dict = {}
    p_feed, pp_list, F_dil_list, split_list = process_pm
    F_rec_list = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
    c_rec_list = [[0]*pm['ns'],[0]*pm['ns'],[0]*pm['ns'],[0]*pm['ns'],[0]*pm['ns'],[0]*pm['ns']]

    def out_of_tolerance(a,a_new):
        if a == 0:
            if np.abs(a_new-a) <= 0.02:
                return False
            else:
                return True
        else:
            if np.abs(a_new-a)/a <= 0.05:
                return False
            else:
                return True
    
    def list_out_of_tolerance(a_list,a_list_new,b_list,b_list_new):
        ret = False
        for i in range(len(a_list)):
            if out_of_tolerance(a_list[i],a_list_new[i]):
                ret = True
                break
        # for i in range(len(b_list)):
        #     for j in range(len(b_list[i])):
        #         if out_of_tolerance(b_list[i][j],b_list_new[i][j]):
        #             ret = True
        #             break
        return ret

    if model_d == False:
        for i in range(len(F_dil_list)):
            F_dil_list[i] = 0
    
    constants = [pm['Li_list'], pm['zi_list'], pm['P1'], pm['P0'], pm['OB2'], pm['OB1'], pm['OB0'], pm['OP1'], pm['OP0'], pm['b0'], pm['h'],pm['rho'],pm['l_module'],pm['eta'],pm['l_mesh'],pm['df'],pm['theta'],pm['n_env'],pm['b_env'],pm['Ki_list'],pm['sf']]

    has_error = True

    counter = 0
    while has_error and counter < 20:
        print('New iter', F_rec_list)

        # 1st stages
        stage_parameters_0 = [0.5 * pm['F_feed'], pm['c_feed'], pm['A'], pm['T'], pm['nk'], pm['nn'], pm['ne'], p_feed, pp_list[0], F_dil_list[0], F_rec_list[0], c_rec_list[0]]
        sol_dict[0] = sdec_r_stage_sim(stage_parameters_0, constants)
        # print(sol_dict[0]['Fr'])

        stage_parameters_2 = [0.5 * pm['F_feed'], pm['c_feed'], pm['A'], pm['T'], pm['nk'], pm['nn'], pm['ne'], p_feed, pp_list[2], F_dil_list[2], F_rec_list[2], c_rec_list[2]]
        sol_dict[2] = sdec_r_stage_sim(stage_parameters_2, constants)
        # print(sol_dict[2]['Fr'])

        # 2nd stages
        stage_parameters_1 = [sol_dict[0]['Fr'], sol_dict[0]['cr'], pm['A'], pm['T'], pm['nk'], pm['nn'], pm['ne'], sol_dict[0]['pr'], pp_list[1], F_dil_list[1], F_rec_list[1], c_rec_list[1]]
        sol_dict[1] = sdec_r_stage_sim(stage_parameters_1, constants)
        # print(sol_dict[1]['Fr'])

        stage_parameters_3 = [sol_dict[2]['Fr'], sol_dict[2]['cr'], pm['A'], pm['T'], pm['nk'], pm['nn'], pm['ne'], sol_dict[2]['pr'], pp_list[3], F_dil_list[3], F_rec_list[3], c_rec_list[3]]
        sol_dict[3] = sdec_r_stage_sim(stage_parameters_3, constants) 
        # print(sol_dict[3]['Fr'])

        # 3rd stage
        F0_4 = sol_dict[1]['Fr'] + sol_dict[3]['Fr']
        c0_4 = []
        for i in range(len(pm['c_feed'])):
            c0_4.append((sol_dict[1]['Fr']*sol_dict[1]['cr'][i] + sol_dict[3]['Fr']*sol_dict[3]['cr'][i])/F0_4)
        
        stage_parameters_4 = [F0_4, c0_4, pm['A'], pm['T'], pm['nk'], pm['nn'], pm['ne'], sol_dict[1]['pr'], pp_list[4], F_dil_list[4],  F_rec_list[4], c_rec_list[4]]
        sol_dict[4] = sdec_r_stage_sim(stage_parameters_4, constants)
        # print(sol_dict[4]['Fr'])

        #4th stage
        stage_parameters_5 = [sol_dict[4]['Fr'], sol_dict[4]['cr'], pm['A'], pm['T'], pm['nk'], pm['nn'], pm['ne'], sol_dict[4]['pr'], pp_list[5], F_dil_list[5], F_rec_list[5], c_rec_list[5]]
        sol_dict[5] = sdec_r_stage_sim(stage_parameters_5, constants)
        # print(sol_dict[5]['Fr'])

        F_rec_list_new = [sol_dict[1]['Fp']*split_list[1], sol_dict[4]['Fp']*split_list[4]*0.5, sol_dict[3]['Fp']*split_list[3], sol_dict[4]['Fp']*split_list[4]*0.5, sol_dict[5]['Fp']*split_list[5], 0]
        c_rec_list_new = [sol_dict[1]['cp'], sol_dict[4]['cp'], sol_dict[3]['cp'], sol_dict[4]['cp'], sol_dict[5]['cp'], [0]*pm['ns']]
        has_error = list_out_of_tolerance(F_rec_list,F_rec_list_new,c_rec_list,c_rec_list_new)
        # print(F_rec_list,F_rec_list_new)
        F_rec_list = F_rec_list_new
        c_rec_list = c_rec_list_new
        counter += 1

    if counter == 20:
        print('Recycle iteration stopped')
    # Performance descriptors
    recovery = (sol_dict[5]['Fr'] * (sol_dict[5]['cr'][0] + sol_dict[5]['cr'][1])) / (pm['F_feed'] * (pm['c_feed'][0]+ pm['c_feed'][1]))
    water_recovery = 1 - (sol_dict[5]['Fr']/(pm['F_feed'] + np.sum(F_dil_list)))
    sep_factor = ((sol_dict[5]['cr'][0] + sol_dict[5]['cr'][1]) / (sol_dict[5]['cr'][2] + sol_dict[5]['cr'][3])) / ((pm['c_feed'][0]+pm['c_feed'][1]) / (pm['c_feed'][2]+pm['c_feed'][3]))
    if model_x:
        p_pump = (pm['F_feed']*p_feed - pm['pex_eff']*sol_dict[5]['Fr']*sol_dict[5]['pr'])/pm['F_feed']
    else: 
        p_pump = p_feed

    dil_power = {}
    for i in range(len(F_dil_list)):
        dil_power[i] = (1/pm['pump_eff']) * F_dil_list[i] * sol_dict[i]['p0']
    
    rec_power = {}
    rec_power[0] = 0
    rec_power[2] = 0
    rec_power[1] = (1/pm['pump_eff']) * sol_dict[1]['Fp'] * split_list[1] * (sol_dict[0]['p0'] - sol_dict[1]['pp'])
    rec_power[3] = (1/pm['pump_eff']) * sol_dict[3]['Fp'] * split_list[3] * (sol_dict[2]['p0'] - sol_dict[3]['pp'])
    rec_power[4] = (1/pm['pump_eff']) * sol_dict[4]['Fp'] * split_list[4] * (sol_dict[1]['p0'] - sol_dict[4]['pp'])
    rec_power[5] = (1/pm['pump_eff']) * sol_dict[5]['Fp'] * split_list[5] * (sol_dict[4]['p0'] - sol_dict[5]['pp'])

    power = (1/pm['pump_eff']) * pm['F_feed'] * p_pump + np.sum(dil_power[i] for i in range(len(F_dil_list))) + np.sum(rec_power[i] for i in range(pm['nst']))

    mol_power = power / recovery
    
    desc = [recovery,water_recovery,sep_factor,p_pump,power,mol_power,dil_power,rec_power]

    return desc, sol_dict