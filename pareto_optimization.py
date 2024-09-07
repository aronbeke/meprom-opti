import opti_multiobjective
import opti_initialize

#################################

# OPTIMIZATION
# Reference sep factor: 5.1 (5.9), recovery : 0.89

no_nodes = 5
no_of_models = 5
dp_max = 40
no_collocation = 6
opti_type = 'obj_sf_con_rec_40bar'
constraint_type = 'recovery'
objective_type = 'separation_factor'
add_rec_const = 0
config = '2d2s'

# constraint_levels = [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 9]
# constraint_levels = [2,2.5,3,3.5,4,4.5,5,5.059,5.142,5.5,6,6.5,7,7.5,8,8.5,9]
# constraint_levels = [2,2.5,3,3.5,4,4.5,5,5.1,5.5,6,6.5,7,7.5,8,8.5,9]
constraint_levels = [0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98]
# constraint_levels = [0.88, 0.89, 0.90, 0.92, 0.96]

model_x = True
model_d = True
model_r = True

opti_multiobjective.multiobjective_optimization_multistart(opti_type,no_of_models,constraint_type,objective_type,constraint_levels,dp_max,no_collocation,no_nodes,config=config,model_x=model_x,model_d=model_d,model_r=model_r,additional_rec_constraint=add_rec_const)


#####################################

# OPTIMIZATION

no_nodes = 5
no_of_models = 5
dp_max = 40
no_collocation = 6
opti_type = 'obj_sp_con_sf_95rec_40bar'
constraint_type = 'separation_factor'
objective_type = 'specific_power'
add_rec_const = 0.95
config = '2d2s'

constraint_levels = [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.1, 5.5, 5.9, 6, 6.5, 7, 7.5, 8, 8.5, 9]

model_x = True
model_d = True
model_r = True

opti_multiobjective.multiobjective_optimization_multistart(opti_type,no_of_models,constraint_type,objective_type,constraint_levels,dp_max,no_collocation,no_nodes,config=config,model_x=model_x,model_d=model_d,model_r=model_r,additional_rec_constraint=add_rec_const)

### VALIDATION

input_file = 'results/opti_sdec_x_d_r_5models_obj_sf_con_rec_40bar_all'
opti_initialize.sim_validation(input_file,40,6,5,model_x=True, model_d=True, model_r = True)

input_file = 'results/opti_sdec_x_d_r_5models_obj_sp_con_sf_95rec_40bar_all'
opti_initialize.sim_validation(input_file,40,6,5,model_x=True, model_d=True, model_r = True)

### PARETO SELECTION

file_name = 'results/opti_sdec_x_d_r_5models_obj_sf_con_rec_40bar_all_validation'
opti_multiobjective.pareto_selector(file_name,'BIVALENT_REC','SEP_FACTOR',competing='true')

file_name = 'results/opti_sdec_x_d_r_5models_obj_sp_con_sf_95rec_40bar_all_validation'
opti_multiobjective.pareto_selector(file_name,'SEP_FACTOR','MOL_POWER',competing='2min')