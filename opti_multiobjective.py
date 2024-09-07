import pandas as pd
import numpy as np
import optimization_sdec as optimization_models
import opti_initialize


def multiobjective_optimization_multistart(opti_type,no_of_models,constraint_type,objective_type,constraint_levels,dp_max,no_collocation,no_nodes,config='2d2s',model_x=False,model_d=False,model_r=False,additional_rec_constraint=0):

    prefix = 'opti_sdec_'
    if model_x:
        prefix += 'x_'
    if model_d:
        prefix += 'd_'
    if model_r:
        prefix += 'r_'

    file_name = prefix+str(no_of_models)+'models_'+opti_type
    pm, pm_sim = opti_initialize.load_swcc_problem(dp_max,no_collocation,no_nodes)

    length = len(constraint_levels)*no_of_models
    WATER_REC = np.zeros(length)
    BIVALENT_REC = np.zeros(length)
    SEP_FACTOR = np.zeros(length)
    MOL_POWER = np.zeros(length)
    FEED_PRESSURE = np.zeros(length)
    PERMEATE_PRESSURES_ARR = [np.zeros(length),np.zeros(length),np.zeros(length),np.zeros(length),np.zeros(length),np.zeros(length)]
    DILUTIONS_ARR = [np.zeros(length),np.zeros(length),np.zeros(length),np.zeros(length),np.zeros(length),np.zeros(length)]
    OMEGA_ARR = [np.zeros(length),np.zeros(length),np.zeros(length),np.zeros(length)]
    OPTIMAL = np.zeros(length)


    i = 0
    for cl in constraint_levels:
        models = []

        init_pm = {
            'p_feed_min' : 10,
            'p_feed_max' : dp_max,
            'dilution_max' : 5,
        }

        for _ in range(no_of_models): 
            print(i)
            model_dict = {}
            if additional_rec_constraint == 0:
                constraints={constraint_type:cl}
            else:
                constraints={constraint_type:cl, 'recovery':additional_rec_constraint}
            objective = objective_type

            # BUILD MODELS
            # model = optimization_models.model_sdec_xrd(constraints,objective,pm,config=config)
            # model = optimization_models.model_sdec(constraints,objective,pm,config=config)
            
            if model_x and model_r and model_d:
                model_5 = optimization_models.model_sdec_xrd(constraints,objective,pm,config=config)
            elif model_r:
                model_5 = optimization_models.model_sdec_r(constraints,objective,pm,config=config)
            elif model_x:
                model_5 = optimization_models.model_sdec_x(constraints,objective,pm,config=config)
            elif model_d:
                model_5 = optimization_models.model_sdec_d(constraints,objective,pm,config=config)
            else:
                model_5 = optimization_models.model_sdec(constraints,objective,pm,config=config)

            # INITIALIZE
            # model, init_parameters = opti_initialize.random_initialization(model,pm_sim,init_pm,model_x=model_x,model_d=model_d,model_r = model_r)
            model_5, init_parameters = opti_initialize.random_initialization(model_5,pm_sim,init_pm,model_x=model_x,model_d=model_d,model_r = model_r)

            # OPTIMIZE AND TRANSFER
            # model, solv_results, optimal = optimization_models.opti(model,solver='ipopt')
            # model, model_5 = opti_initialize.variable_value_transfer(model,model_5,pm['ns'],pm['nk'],pm['nn'],pm['ne'],pm['nst'],model_x=model_x,model_d=model_d,model_r = model_r)
            model_5, solv_results_5, optimal_5 = optimization_models.opti(model_5,solver='ipopt')

            model_dict['init'] = {}
            model_dict['init']['p_feed'] = init_parameters[0]
            model_dict['init']['pp_list'] = init_parameters[1]
            model_dict['init']['F_dil_list'] = init_parameters[2]
            model_dict['optimal'] = optimal_5
            model_dict['mol_power'] = model_5.mol_power.value
            model_dict['recovery'] = model_5.recovery.value
            model_dict['water_recovery'] = model_5.water_recovery.value
            model_dict['separation_factor'] = model_5.separation_factor.value
            model_dict['dilutions'] = {}
            model_dict['split_ratios'] = {}
            if model_x:
                model_dict['feed_pressure'] = model_5.p_pex.value
            else:
                model_dict['feed_pressure'] = model_5.p_feed.value
            model_dict['permeate_pressures'] = {}
            for j in range(pm['nst']):
                if model_d:
                    model_dict['dilutions'][j] = model_5.F_dilution[j].value
                else:
                    model_dict['dilutions'][j] = 0
                if model_r:
                    model_dict['split_ratios'][j] = model_5.omega[j].value
                else:
                    model_dict['split_ratios'][j] = 0
                model_dict['permeate_pressures'][j] = model_5.stages[j].pp.value

            OPTIMAL[i] = optimal_5
            BIVALENT_REC[i] = model_5.recovery.value
            WATER_REC[i] = model_5.water_recovery.value
            SEP_FACTOR[i] = model_5.separation_factor.value
            MOL_POWER[i] = model_5.mol_power.value
            FEED_PRESSURE[i] = model_dict['feed_pressure']
            OMEGA_ARR[0][i] = model_dict['split_ratios'][1]
            OMEGA_ARR[1][i] = model_dict['split_ratios'][3]
            OMEGA_ARR[2][i] = model_dict['split_ratios'][4]
            OMEGA_ARR[3][i] = model_dict['split_ratios'][5]
            for j in range(pm['nst']):
                DILUTIONS_ARR[j][i] = model_dict['dilutions'][j]
                PERMEATE_PRESSURES_ARR[j][i] = model_dict['permeate_pressures'][j]
            
            models.append(model_dict)
            i += 1


    res_data_all = {
        'OPTIMAL': OPTIMAL,
        'WATER_REC': WATER_REC,
        'BIVALENT_REC': BIVALENT_REC,
        'SEP_FACTOR': SEP_FACTOR,
        'MOL_POWER': MOL_POWER,
        'FEED_PRESSURE': FEED_PRESSURE,
        'OMEGA_1': OMEGA_ARR[0],
        'OMEGA_3': OMEGA_ARR[1],
        'OMEGA_4': OMEGA_ARR[2],
        'OMEGA_5': OMEGA_ARR[3],
        'DILUTION_0': DILUTIONS_ARR[0],
        'DILUTION_1': DILUTIONS_ARR[1],
        'DILUTION_2': DILUTIONS_ARR[2],
        'DILUTION_3': DILUTIONS_ARR[3],
        'DILUTION_4': DILUTIONS_ARR[4],
        'DILUTION_5': DILUTIONS_ARR[5],
        'PERMEATE_PRESSURE_0': PERMEATE_PRESSURES_ARR[0],
        'PERMEATE_PRESSURE_1': PERMEATE_PRESSURES_ARR[1],
        'PERMEATE_PRESSURE_2': PERMEATE_PRESSURES_ARR[2],
        'PERMEATE_PRESSURE_3': PERMEATE_PRESSURES_ARR[3],
        'PERMEATE_PRESSURE_4': PERMEATE_PRESSURES_ARR[4],
        'PERMEATE_PRESSURE_5': PERMEATE_PRESSURES_ARR[5],
    }


    results_all_df = pd.DataFrame(res_data_all)

    results_all_df.to_csv('results/'+file_name+"_all.csv")


def pareto_selector(file_name_without_csv,objective1,objective2,competing='true'):
    '''
    competing: 
    true -> classical pareto trade-off
    1min -> minimizing objective 1, maximizing objective 2
    2min -> minimizing objective 2, maximizing objective 1
    '''

    # Step 1: Read the CSV file
    input_file_path = file_name_without_csv+'.csv'
    output_file_path = file_name_without_csv+'_pareto.csv'

    df = pd.read_csv(input_file_path)

    # Step 2: Identify Pareto-optimal records based on objectives A and B
    pareto_optimal = []
    pareto_optimal_points = []

    for index, row in df.iterrows():
        current_point = (round(row[objective1],4), round(row[objective2],4))
        is_pareto_optimal = True
        is_feasible = True
        if np.abs(row['SIM_SEP_FACTOR'] - row['SEP_FACTOR'])/row['SIM_SEP_FACTOR'] > 0.1:
            is_feasible = False
        for idx, rw in df.iterrows():
            current_point2 = (round(rw[objective1],4), round(rw[objective2],4))
            if competing == 'true':
                if (current_point2[0] >= current_point[0] and current_point2[1] >= current_point[1] and (current_point2[0] != current_point[0] or current_point2[1] != current_point[1])) or current_point in pareto_optimal_points:
                    is_pareto_optimal = False
                    break
            elif competing == '1min':
                if (current_point2[0] <= current_point[0] and current_point2[1] >= current_point[1] and (current_point2[0] != current_point[0] or current_point2[1] != current_point[1])) or current_point in pareto_optimal_points:
                    is_pareto_optimal = False
                    break
            elif competing == '2min':
                if (current_point2[0] >= current_point[0] and current_point2[1] <= current_point[1] and (current_point2[0] != current_point[0] or current_point2[1] != current_point[1])) or current_point in pareto_optimal_points:
                    is_pareto_optimal = False
                    break
        if is_pareto_optimal and row['OPTIMAL'] == 1 and is_feasible:
            pareto_optimal.append(row)
            pareto_optimal_points.append(current_point)

    pareto_df = pd.DataFrame(pareto_optimal)

    # Step 3: Save the Pareto-optimal records as another CSV file
    pareto_df.to_csv(output_file_path, index=False)