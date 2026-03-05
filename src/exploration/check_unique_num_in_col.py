import os
import pandas as pd

categorical_cols = ['piezo_station_department_code',
                    'piezo_station_department_name', 'piezo_station_commune_code_insee',
                    'piezo_station_pe_label',
                    'piezo_station_bss_code', 'piezo_station_commune_name',
                    'piezo_station_bss_id', 'piezo_bss_code',
                    'piezo_obtention_mode', 'piezo_status', 'piezo_qualification',
                    'piezo_continuity_name', 'piezo_producer_name',
                    'piezo_measure_nature_code', 'piezo_measure_nature_name', 'meteo_name',
                    'hydro_station_code',
                    'hydro_status_label', 'hydro_method_label', 'hydro_qualification_label',
                    'hydro_hydro_quantity_elab', 'prelev_structure_code_0',
                    'prelev_usage_label_0', 'prelev_volume_obtention_mode_label_0',
                    'prelev_structure_code_1', 'prelev_usage_label_1',
                    'prelev_volume_obtention_mode_label_1', 'prelev_structure_code_2',
                    'prelev_usage_label_2', 'prelev_volume_obtention_mode_label_2',
                    'piezo_groundwater_level_category']

multi_values_cols = ['piezo_station_bdlisa_codes']

col = categorical_cols[0]

csv_dir = "./outputs/check_na"
unique_num_dict = {}

for col in categorical_cols + multi_values_cols:
    csv_filename = os.path.join(csv_dir, col+".csv")
    df = pd.read_csv(csv_filename)
    unique_num_dict[col] = len(df)