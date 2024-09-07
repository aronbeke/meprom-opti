import pandas as pd

def read_input(input_file_path):
    '''
    Reads inputs provided in .txt file.
    '''
    data_dict = {}
    with open(input_file_path, 'r') as file:
        for line in file:
            # Split each line by the first '=' character
            key, value = line.strip().split('=', 1)
            # Convert numerical values to float
            try:
                value = float(value)
            except ValueError:
                pass
            data_dict[key] = value
    
    data_dict['pressure_exchange'] = int(data_dict['pressure_exchange'])
    data_dict['interstage_dilution'] = int(data_dict['interstage_dilution'])
    data_dict['permeate_recycling'] = int(data_dict['permeate_recycling'])
    return data_dict

def check_input_data(data_dict):
    error = False
    error_message = ''
    if data_dict['objective'] not in ('separation_factor','molar_power'):
        error = True
        error_message = 'Unknown objective.'
        return error, error_message

    if data_dict['maximum_pressure'] > 50 or data_dict['maximum_pressure'] < 10:
        error = True
        error_message = 'Feed pressure has to be between 10 and 50 barg.'
        return error, error_message
    
    if data_dict['no_models'] > 20 or data_dict['no_models'] < 1:
        error = True
        error_message = 'Number of models has to be between 1 and 20.'
        return error, error_message
    
    if data_dict['objective'] == 'separation_factor' and (data_dict['min_constraint'] < 0.0 or  data_dict['min_constraint'] > 1.0):
        error = True
        error_message = 'Recovery constraint for separation factor optimization has to be between 0 and 1.'
        return error, error_message

    if data_dict['objective'] == 'separation_factor' and (data_dict['max_constraint'] < 0.0 or  data_dict['max_constraint'] > 1.0):
        error = True
        error_message = 'Recovery constraint for separation factor optimization has to be between 0 and 1.'
        return error, error_message

    if data_dict['objective'] == 'molar_power' and (data_dict['min_constraint'] < 2.0 or  data_dict['min_constraint'] > 10.0):
        error = True
        error_message = 'Separation factor constraint for molar power optimization has to be between 2 and 10.'
        return error, error_message

    if data_dict['objective'] == 'molar_power' and (data_dict['max_constraint'] < 2.0 or  data_dict['max_constraint'] > 10.0):
        error = True
        error_message = 'Separation factor constraint for molar power optimization has to be between 2 and 10.'
        return error, error_message
    
    if data_dict['max_constraint'] < data_dict['min_constraint']:
        error = True
        error_message = 'Maximum constraint has to be higher or equal to minimum constraint.'
        return error, error_message

    if data_dict['pressure_exchange'] != 1 or data_dict['pressure_exchange'] != 0:
        error = True
        error_message = 'Pressure exchange value has to be 1 or 0.'
        return error, error_message

    if data_dict['interstage_dilution'] != 1 or data_dict['interstage_dilution'] != 0:
        error = True
        error_message = 'Interstage dilution value has to be 1 or 0.'
        return error, error_message            

    if data_dict['permeate_recycling'] != 1 or data_dict['permeate_recycling'] != 0:
        error = True
        error_message = 'Permeate recycling value has to be 1 or 0.'
        return error, error_message                                

    return error, error_message