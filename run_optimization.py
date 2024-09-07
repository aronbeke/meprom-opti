import opti_multiobjective
import opti_initialize
import auxiliary
import numpy as np

if __name__ == "__main__":

    input_path = input('Input file path or name: ')
    if not input_path.endswith(".txt"):
        input_path += ".txt"

    try:
        data = auxiliary.read_input(input_path)
    except FileNotFoundError:
        print('File not found. All input data has to be provided in an input .txt file.')
        exit()

    is_error, error_message = auxiliary.check_input_data(data)
    if is_error:
        print(error_message)
        exit()
    
    no_nodes = data['no_nodes']
    no_of_models = data['no_models']
    dp_max = data['max_pressure']
    no_collocation = 6

    objective_type = data['objective']

    if objective_type == 'separation_factor':
        constraint_type = 'recovery'
        opti_type = 'obj_sf_con_rec'
    elif objective_type == 'molar_power':
        constraint_type = 'separation_factor'
        opti_type = 'obj_mp_con_sf'

    add_rec_const = 0
    config = '2d2s'

    if data['pressure_exchange'] == 1:
        model_x = True
    elif data['pressure_exchange'] == 0:
        model_x = False
    
    if data['interstage_dilution'] == 1:
        model_d = True
    elif data['interstage_dilution'] == 0:
        model_d = False

    if data['permeate_recycling'] == 1:
        model_r = True
    elif data['permeate_recycling'] == 0:
        model_r = False

    constraint_levels = data['constraint_list']

    prefix = 'opti_sdec_'
    if model_x:
        prefix += 'x_'
    if model_d:
        prefix += 'd_'
    if model_r:
        prefix += 'r_'

    file_name = prefix+str(no_of_models)+'models_'+opti_type
    input_path = 'results/'+file_name

    print("STARTING MULTISTART OPTIMIZATION\n")
    opti_multiobjective.multiobjective_optimization_multistart(opti_type,no_of_models,constraint_type,objective_type,constraint_levels,dp_max,no_collocation,no_nodes,config=config,model_x=model_x,model_d=model_d,model_r=model_r,additional_rec_constraint=add_rec_const)

    print("STARTING VALIDATION THROUGH PROCESS SIMULATION\n")
    # Validation
    input_file = input_path+'_all'
    opti_initialize.sim_validation(input_file,dp_max,no_collocation,no_nodes,model_x=model_x, model_d=model_d, model_r = model_r)

    print("STARTING PARETO SELECTION")
    # Pareto selection
    input_file = input_path+'_all_validation'
    if objective_type == 'separation_factor':
        opti_multiobjective.pareto_selector(input_file,'DIVALENT_REC','SEP_FACTOR',competing='true')
    if objective_type == 'molar_power':
        opti_multiobjective.pareto_selector(input_file,'SEP_FACTOR','MOL_POWER',competing='2min')
    print("TERMINATION SUCCESSFUL")
   