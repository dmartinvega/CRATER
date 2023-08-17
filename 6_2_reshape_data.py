from pickle import dump

import numpy as np
import pandas as pd
import importlib
train_test_split = importlib.reload(__import__('6_1_train_test_split'))


def get_set_from_ids(ids, classification_matrix_filename):
    path_df_class_matrix = '/home/daniela/Documents/Academico/Maestria/astro_transient_class_css/' + classification_matrix_filename
    df_class_matrix = pd.read_csv(path_df_class_matrix, error_bad_lines=False)
    return df_class_matrix[df_class_matrix.ID.isin(ids)]


def reshape_set(set, classification_columns, samples_per_lightcurve):
    classification_columns = ["Classification_" + event for event in classification_columns]

    number_of_lightcurves = set.shape[0] / samples_per_lightcurve
    ids = set.ID.unique()
    tensor_x = np.zeros((int(number_of_lightcurves), samples_per_lightcurve, 1))
    tensor_y = np.zeros((int(number_of_lightcurves), len(classification_columns), samples_per_lightcurve))

    for i, id in enumerate(ids):
        tensor_x[i, :, :] = set[set.ID == id]['Flux'][:, np.newaxis]
        tensor_y[i, :, :] = np.transpose(set[set.ID == id][classification_columns])

    return tensor_x, tensor_y


def save_reshaped_sets(tensor_x, tensor_y, x_name, y_name):
    dump(tensor_x, open('data/preprocessed/' + x_name + '_reshaped.pkl', 'wb'))
    dump(tensor_y, open('data/preprocessed/' + y_name + '_reshaped.pkl', 'wb'))


def main(x_train, y_train, x_test, y_test, samples_per_lightcurve, classification_columns,
         classification_matrix_filename='data/raw/classification_matrix.csv'):
    x_train_ids, x_test_ids = train_test_split.open_training_and_test_sets(x_train, x_test)
    train_set = get_set_from_ids(x_train_ids, classification_matrix_filename)
    test_set = get_set_from_ids(x_test_ids, classification_matrix_filename)

    x_train_set_reshaped, y_train_set_reshaped = reshape_set(train_set, classification_columns, samples_per_lightcurve)
    save_reshaped_sets(x_train_set_reshaped, y_train_set_reshaped, x_train, y_train)
    x_test_set_reshaped, y_test_set_reshaped = reshape_set(test_set, classification_columns, samples_per_lightcurve)
    save_reshaped_sets(x_test_set_reshaped, y_test_set_reshaped, x_test, y_test)
