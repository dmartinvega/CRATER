import statistics
from math import floor, ceil
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from tqdm import tqdm


def main(transient_id):
    df_transient_lightcurves, df_transients_days_before_and_after_trigger = read_csvs_lightcurves_and_classification()

    save_days_before_and_after_trigger_in_transient_classification_all_events(df_transient_lightcurves,
                                                                              df_transients_days_before_and_after_trigger)

    df_transient_lightcurve = df_transient_lightcurves[df_transient_lightcurves.ID == transient_id]

    f = interpolate_linearly_lightcurves(df_transient_lightcurve)
    return create_input_vector(f, transient_id, df_transients_days_before_and_after_trigger, df_transient_lightcurve)


def all_events():
    df_interpolated_lightcurves = pd.DataFrame(columns=['ID', 'MJD', 'Flux'])
    df_transient_lightcurves, df_transients_days_before_and_after_trigger = read_csvs_lightcurves_and_classification()

    for index, row in tqdm(df_transients_days_before_and_after_trigger.iterrows()):
        save_days_before_and_after_trigger_in_transient_classification_all_events(df_transient_lightcurves,
                                                                                  df_transients_days_before_and_after_trigger,
                                                                                  index, row)

        df_transient_lightcurve = df_transient_lightcurves[df_transient_lightcurves.ID == row['TranID']]

        f = interpolate_linearly_lightcurves(df_transient_lightcurve)
        interpolated_lightcurve_individual = create_input_vector(f, row['TranID'], df_transients_days_before_and_after_trigger, df_transient_lightcurve)
        df_interpolated_lightcurves = df_interpolated_lightcurves.append(interpolated_lightcurve_individual, ignore_index=True)

    df_interpolated_lightcurves.to_csv('data/raw/transient_lightcurves_interpolated_150_days.csv')

def save_days_before_and_after_trigger_in_transient_classification_all_events(df_transient_lightcurves,
                                                                              df_transients_days_before_and_after_trigger,
                                                                              index, row):
    df_transient = df_transient_lightcurves[df_transient_lightcurves.ID == row['TranID']]
    min_mjd = df_transient['MJD'].min(skipna=True)
    max_mjd = df_transient['MJD'].max(skipna=True)
    trigger_series = df_transient.dropna(subset=['TriggerFluxNeg3'])['MJD']
    trigger = trigger_series[trigger_series.index[0]]
    df_transients_days_before_and_after_trigger.loc[index, 'DaysBeforeTrigger'] = trigger - min_mjd
    df_transients_days_before_and_after_trigger.loc[index, 'DaysAfterTrigger'] = max_mjd - trigger


def save_days_before_and_after_trigger_in_transient_classification(df_transient_lightcurves,
                                                                   df_transients_days_before_and_after_trigger,
                                                                   transient_id):
    for index, row in tqdm(df_transients_days_before_and_after_trigger.iterrows()):
        if row['TranID'] == transient_id:
            df_transient = df_transient_lightcurves[df_transient_lightcurves.ID == row['TranID']]
            min_mjd = df_transient['MJD'].min(skipna=True)
            max_mjd = df_transient['MJD'].max(skipna=True)
            trigger_series = df_transient.dropna(subset=['TriggerFluxNeg3'])['MJD']
            trigger = trigger_series[trigger_series.index[0]]
            df_transients_days_before_and_after_trigger.loc[index, 'DaysBeforeTrigger'] = trigger - min_mjd
            df_transients_days_before_and_after_trigger.loc[index, 'DaysAfterTrigger'] = max_mjd - trigger


def read_csvs_lightcurves_and_classification():
    path_df_transient_lightcurves = '/home/daniela/Documents/Academico/Maestria/astro_transient_class_css/data/raw/transient_lightcurves_selected_fluxneg_3sigma.csv'
    df_transient_lightcurves = pd.read_csv(path_df_transient_lightcurves, error_bad_lines=False)
    path_df_transient_classification = '/home/daniela/Documents/Academico/Maestria/astro_transient_class_css/data/raw/classification_transient_lightcurves_triggered_selected.csv'
    df_transients_days_before_and_after_trigger = pd.read_csv(path_df_transient_classification, error_bad_lines=False)
    return df_transient_lightcurves, df_transients_days_before_and_after_trigger


def interpolate_linearly_lightcurves(df_transient_lightcurve):
    x = df_transient_lightcurve['MJD']
    y = df_transient_lightcurve['FluxNeg']
    f = interp1d(x, y)
    return f


def create_input_vector(f, transient_id, df_transients_days_before_and_after_trigger, df_transient_lightcurve):
    days_after_trigger, days_before_trigger = get_days_before_and_after_trigger(transient_id,
                                                                                df_transients_days_before_and_after_trigger)

    days_after_left, days_before_left, t_trigger = get_days_remaining_to_complete_input_vector(days_after_trigger,
                                                                                               days_before_trigger,
                                                                                               df_transient_lightcurve)

    incomplete_after_days, incomplete_before_days, lower_limit, upper_limit = find_if_is_incomplete_before_or_after_trigger(
        days_after_left, days_after_trigger, days_before_left, days_before_trigger, t_trigger)

    number_of_samples = get_number_of_samples_real(days_after_trigger, days_before_trigger, incomplete_after_days,
                                                   incomplete_before_days)

    x_input_vector, y_input_vector = create_input_vector_real(f, lower_limit, number_of_samples, upper_limit)

    if number_of_samples < 150:
        x_input_vector, y_input_vector = complete_vector_if_needed(days_after_left, days_before_left,
                                                                   incomplete_after_days,
                                                                   incomplete_before_days, number_of_samples, t_trigger,
                                                                   y_input_vector)

    return convert_input_vector_to_df(transient_id, x_input_vector, y_input_vector)


def get_days_before_and_after_trigger(transient_id, df_transients_days_before_and_after_trigger):
    df_days_before_and_after_trigger = df_transients_days_before_and_after_trigger[
        df_transients_days_before_and_after_trigger['TranID'] == transient_id]
    days_before_trigger_series = df_days_before_and_after_trigger['DaysBeforeTrigger']
    days_before_trigger = days_before_trigger_series[days_before_trigger_series.index[0]]
    days_after_trigger_series = df_days_before_and_after_trigger['DaysAfterTrigger']
    days_after_trigger = days_after_trigger_series[days_after_trigger_series.index[0]]
    return days_after_trigger, days_before_trigger


def get_days_remaining_to_complete_input_vector(days_after_trigger, days_before_trigger, df_transient_lightcurve):
    t_trigger_series = df_transient_lightcurve.dropna(subset=['TriggerFluxNeg3'])['MJD']
    t_trigger = t_trigger_series[t_trigger_series.index[0]]
    days_before_left = 70 - round(days_before_trigger)
    days_after_left = 80 - round(days_after_trigger)
    return days_after_left, days_before_left, t_trigger


def find_if_is_incomplete_before_or_after_trigger(days_after_left, days_after_trigger, days_before_left,
                                                  days_before_trigger, t_trigger):
    incomplete_before_days, lower_limit = find_if_is_incomplete_before_trigger(days_before_left, days_before_trigger,
                                                                               t_trigger)
    incomplete_after_days, upper_limit = find_if_is_incomplete_after_trigger(days_after_left, days_after_trigger,
                                                                             t_trigger)
    return incomplete_after_days, incomplete_before_days, lower_limit, upper_limit


def find_if_is_incomplete_before_trigger(days_before_left, days_before_trigger, t_trigger):
    if days_before_left > 0:
        lower_limit = ceil(t_trigger - days_before_trigger)
        incomplete_before_days = True
    else:
        lower_limit = ceil(t_trigger - 70)
        incomplete_before_days = False
    return incomplete_before_days, lower_limit


def find_if_is_incomplete_after_trigger(days_after_left, days_after_trigger, t_trigger):
    if days_after_left > 0:
        upper_limit = floor(t_trigger + days_after_trigger)
        incomplete_after_days = True
    else:
        upper_limit = floor(t_trigger + 80)
        incomplete_after_days = False
    return incomplete_after_days, upper_limit


def get_number_of_samples_real(days_after_trigger, days_before_trigger, incomplete_after_days, incomplete_before_days):
    if incomplete_before_days and incomplete_after_days:
        number_of_samples = round(days_before_trigger + days_after_trigger)
    elif incomplete_before_days and not incomplete_after_days:
        number_of_samples = round(days_before_trigger + 80)
    elif not incomplete_before_days and incomplete_after_days:
        number_of_samples = round(70 + days_after_trigger)
    else:
        number_of_samples = 150
    return number_of_samples


def create_input_vector_real(f, lower_limit, number_of_samples, upper_limit):
    x_input_vector = np.linspace(lower_limit, upper_limit, num=number_of_samples, endpoint=True)
    y_input_vector = f(x_input_vector)
    return x_input_vector, y_input_vector


def complete_vector_if_needed(days_after_left, days_before_left, incomplete_after_days, incomplete_before_days,
                              number_of_samples, t_trigger, y_input_vector):
    x_input_vector = np.linspace(ceil(t_trigger) - 70, floor(t_trigger) + 80, num=150, endpoint=True)
    complete_vector = np.zeros(150)

    median_before = statistics.median(y_input_vector[0:3])
    median_after = statistics.median(y_input_vector[-3:])

    if incomplete_before_days and incomplete_after_days:
        complete_vector[0:70] = median_before
        complete_vector[70:] = median_after
        complete_vector[days_before_left:days_before_left + number_of_samples] = y_input_vector
        y_input_vector = complete_vector
    elif incomplete_before_days:
        complete_vector[0:70] = median_before
        complete_vector[days_before_left:] = y_input_vector
        y_input_vector = complete_vector
    elif incomplete_after_days:
        complete_vector[70:] = median_after
        complete_vector[:-days_after_left] = y_input_vector
        y_input_vector = complete_vector
    return x_input_vector, y_input_vector


def convert_input_vector_to_df(transient_id, x_input_vector, y_input_vector):
    tran_id_list = [transient_id] * 150
    result = zip(tran_id_list, x_input_vector, y_input_vector)
    input_vector_set = set(result)
    input_vector_list = list(input_vector_set)
    input_vector_ordered_list = sorted(input_vector_list, key=lambda tup: tup[1])
    return pd.DataFrame(input_vector_ordered_list, columns=['ID', 'MJD', 'Flux'])

# main('TranID906221090914122502')  # 37 izq y der
# main('TranID1204191180784142832')  # > 150
# main('TranID1607081520484130475')  # > 150
# main('TranID1504291690384116269')  # 81 der
# main('TranID1305131120734140944')  # > 150
# main('TranID1101140070504144585')  # 80 izq
# main('TranID909160120034110694')  # > 150
# main('TranID1001121180684103058')  # 118 izq
# main('TranID1109231260224137668')

all_events()
