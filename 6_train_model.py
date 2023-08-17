import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pickle import dump
import importlib
import ast
import importlib

train_test_split_css = importlib.reload(__import__('6_1_train_test_split'))
reshape_data = importlib.reload(__import__('6_2_reshape_data'))
run_model = importlib.reload(__import__('6_3_model'))
train_model_german = importlib.reload(__import__('6_train_model_german'))


# def load_data():
#     X = pd.read_csv("./data/raw/Datos german/means.txt", sep=" ", header=None)
#     y = pd.read_csv("./data/raw/Datos german/labs.txt", header=None)
#     return X, y
#
#
# def select_from_classes(classes, X, y):
#     y = y[y.isin(classes)].dropna()
#     X = X.loc[y.index, :]
#     return X, y


# def subset_observations(X, subset_number):
#     length_transient_lightcurves_columns = len(X.columns)
#
#     if subset_number < length_transient_lightcurves_columns:
#         df_transient_lightcurves_classification_without_label_limit_cut_lower = int(
#             (length_transient_lightcurves_columns / 2) - (subset_number / 2) - 1)
#         df_transient_lightcurves_classification_without_label_limit_cut_upper = int(
#             (length_transient_lightcurves_columns / 2) + (subset_number / 2) - 1)
#         X = X.iloc[:,
#             df_transient_lightcurves_classification_without_label_limit_cut_lower:df_transient_lightcurves_classification_without_label_limit_cut_upper]
#
#     return X


# def encode_classification(X, y):
#     df_transient_lightcurves_classification = X.copy()
#     df_transient_lightcurves_classification["Classification"] = y
#
#     train_test_split_css = importlib.reload(__import__('6_1_train_test_split'))
#     df_transient_lightcurves_classification_label_encoder = train_test_split_css.save_label_encoder(
#         df_transient_lightcurves_classification)
#
#     class_label_encoded = df_transient_lightcurves_classification_label_encoder['ClassLabelEncoded']
#     class_label_encoded = pd.get_dummies(class_label_encoded)
#
#     return class_label_encoded


def undersample_training_set(x_train_imbalanced, y_train_imbalanced, undersampling_mode):
    x_balanced = x_train_imbalanced
    y_balanced = y_train_imbalanced

    if undersampling_mode == 'equal':
        min_class_representation = y_train_imbalanced.value_counts().min()
        unique_serie = y_train_imbalanced.value_counts().index.to_frame()
        x_balanced = pd.DataFrame()
        y_balanced = pd.DataFrame()
        for row in unique_serie.iterrows():
            df = y_train_imbalanced.reset_index(drop=True).reset_index()
            indice = df[df["ClassLabelEncoded"] == row[0]].index
            lista = np.random.choice(indice, size=min_class_representation)
            y_balanced = pd.concat([y_balanced, y_train_imbalanced.iloc[lista]])
            y_balanced = y_balanced.astype('int64')
            x_balanced = pd.concat([x_balanced, x_train_imbalanced.iloc[lista]])

    if undersampling_mode == 'sn 50 cv 60':
        y_train_sn_only = y_train_imbalanced[y_train_imbalanced == 8]  # 8 = SN
        y_train_sn_50 = y_train_sn_only.sample(n=int(len(y_train_sn_only) / 2))
        y_train_sn_balanced_cv_imbalanced = pd.concat([y_train_imbalanced[y_train_imbalanced != 8],
                                                       y_train_sn_50], axis=0, join="inner")

        y_train_cv_only = y_train_imbalanced[y_train_imbalanced == 3]  # 3 = CV
        y_train_sn_60 = y_train_cv_only.sample(n=int(len(y_train_cv_only) * 0.61))
        y_balanced = pd.concat(
            [y_train_sn_balanced_cv_imbalanced[y_train_sn_balanced_cv_imbalanced != 3],
             y_train_sn_60], axis=0, join="inner")
        x_balanced = x_train_imbalanced.loc[y_balanced.index]

    return x_balanced.squeeze(), y_balanced.squeeze()


def save_train_and_test_data(x_train, x_test, y_train_reshaped, y_test_reshaped, x_train_filename,
                             x_test_filename, y_train_filename, y_test_filename):
    dump(x_train, open('data/preprocessed/' + x_train_filename + '.pkl', 'wb'))
    dump(x_test, open('data/preprocessed/' + x_test_filename + '.pkl', 'wb'))
    dump(y_train_reshaped, open('data/preprocessed/' + y_train_filename + '.pkl', 'wb'))
    dump(y_test_reshaped, open('data/preprocessed/' + y_test_filename + '.pkl', 'wb'))


def model_training(setting):
    df_transient_lightcurves_classification = train_test_split_css.read_csv_classification_lightcurves()
    df_transient_lightcurves_classification_label_encoder = train_test_split_css.save_label_encoder(
        df_transient_lightcurves_classification)

    df_transient_lightcurves_classification_label_encoder = df_transient_lightcurves_classification_label_encoder[
        df_transient_lightcurves_classification_label_encoder['Classification'].isin(
            setting["z_classes"])]

    train_test_split_css.prepare_and_save_sets(df_transient_lightcurves_classification_label_encoder['TranID'],
                                               df_transient_lightcurves_classification_label_encoder[
                                                   'ClassLabelEncoded'], setting["z_train_split"])
    x_train, x_test, y_train, y_test = train_test_split_css.open_training_and_test_sets(setting["z_x_train"],
                                                                                        setting["z_x_test"],
                                                                                        setting["z_y_train"],
                                                                                        setting["z_y_test"])

    if setting["z_undersampling_mode"] is not None:
        x_train, y_train = undersample_training_set(x_train, y_train, setting["z_undersampling_mode"])
        dump(x_train, open('data/preprocessed/x_train.pkl', 'wb'))
        dump(y_train, open('data/preprocessed/y_train.pkl', 'wb'))

    if setting["z_pre_transient"]:
        reshape_data.main(setting["z_x_train"], setting["z_y_train"], setting["z_x_test"], setting["z_y_test"],
                          setting["z_subset_observations"], setting["z_classes"])
    else:
        reshape_data.main(setting["z_x_train"], setting["z_y_train"], setting["z_x_test"], setting["z_y_test"],
                          setting["z_subset_observations"], setting["z_classes"],
                          setting["z_classification_matrix_filename"])

    # df_transient_lightcurves_classification_without_label, y = load_data()
    #
    # if "z_classes" in setting:
    #     df_transient_lightcurves_classification_without_label, y = select_from_classes(setting["z_classes"],
    #                                                                                    df_transient_lightcurves_classification_without_label,
    #                                                                                    y)
    # df_transient_lightcurves_classification_without_label = subset_observations(
    #     df_transient_lightcurves_classification_without_label, setting["z_subset_observations"])

    # class_label_encoded = encode_classification(df_transient_lightcurves_classification_without_label, y)
    #
    # df_transient_lightcurves_classification_without_label_reshaped = df_transient_lightcurves_classification_without_label.to_numpy()[
    #                                                                  :, :, np.newaxis]
    #
    # x_train, x_test, y_train, y_test = train_test_split(
    #     df_transient_lightcurves_classification_without_label_reshaped, class_label_encoded,
    #     train_size=setting["z_train_split"],
    #     test_size=setting["z_test_split"], random_state=42)

    # if "z_undersampling_mode" in setting:
    #     x_train, y_train = undersample_training_set(x_train, y_train, setting["z_undersampling_mode"])
    #
    # y_train_reshaped = np.tile(y_train.to_numpy()[:, :, np.newaxis], reps=setting["z_subset_observations"])
    # y_test_reshaped = np.tile(y_test.to_numpy()[:, :, np.newaxis], reps=setting["z_subset_observations"])
    #
    # save_train_and_test_data(x_train, x_test, y_train_reshaped, y_test_reshaped,
    #                          setting["z_x_train_name"], setting["z_x_test_name"], setting["z_y_train_name"],
    #                          setting["z_y_test_name"])

    run_model.main(setting)


def run_experiments_massively_from_settings(experiments_settings, epochs_list=None):
    for experiment_integer_index in range(0, experiments_settings.shape[0]):
        if not experiments_settings.iloc[experiment_integer_index]['z_already_ran']:
            setting = experiments_settings.iloc[experiment_integer_index].to_dict()
            setting = prepare_setting_for_experiment_when_obtained_from_mlflow(setting)
            original_run_name = setting['z_run_name']
            if epochs_list is None:
                if "rapid" in setting["z_run_name"]:
                    model_training(setting)
                elif "german" in setting["z_run_name"]:
                    train_model_german.model_training_with_german_input_data(setting)
            else:
                for epochs in epochs_list:
                    setting['z_run_name'] = f"{original_run_name}, {epochs} epochs"
                    setting['z_epochs'] = epochs
                    if "rapid" in setting["z_run_name"]:
                        model_training(setting)
                    elif "german" in setting["z_run_name"]:
                        train_model_german.model_training_with_german_input_data(setting)


def prepare_setting_for_experiment_when_obtained_from_mlflow(setting):
    # El DF de experimentos que entrega MLFlow toma todas las columnas como tipo object (string), entonces se deben hacer los cambios a los tipos de datos correctos
    if setting['z_base_lr'] == "None":
        setting['z_base_lr'] = None
    elif setting['z_base_lr'] is None:
        pass
    else:
        setting['z_base_lr'] = float(setting['z_base_lr'])

    if setting['z_learning_rate'] == "None":
        setting['z_learning_rate'] = None
    elif setting['z_learning_rate'] is None:
        pass
    else:
        setting['z_learning_rate'] = float(setting['z_learning_rate'])

    setting['z_batch_size'] = int(setting['z_batch_size'])
    setting['z_bidirectional'] = ast.literal_eval(setting['z_bidirectional'])
    setting['z_classes'] = ast.literal_eval(setting['z_classes'])

    if setting['z_decay_steps'] == "None":
        setting['z_decay_steps'] = None
    elif setting['z_decay_steps'] is None:
        pass
    else:
        setting['z_decay_steps'] = int(setting['z_decay_steps'])

    setting['z_dropout_rate'] = float(setting['z_dropout_rate'])

    if setting['z_end_lr'] == "None":
        setting['z_end_lr'] = None
    elif setting['z_end_lr'] is None:
        pass
    else:
        setting['z_end_lr'] = float(setting['z_end_lr'])

    setting['z_epochs'] = int(setting['z_epochs'])
    setting['z_gru_size_1'] = int(setting['z_gru_size_1'])
    setting['z_gru_size_2'] = int(setting['z_gru_size_2'])

    if setting['z_power'] == "None":
        setting['z_power'] = None
    elif setting['z_power'] is None:
        pass
    else:
        setting['z_power'] = float(setting['z_power'])

    setting['z_pre_transient'] = ast.literal_eval(setting['z_pre_transient'])
    setting['z_return_sequences'] = ast.literal_eval(setting['z_return_sequences'])

    if setting['z_sample_weights'] == "None":
        setting['z_sample_weights'] = None
    elif setting['z_sample_weights'] is None:
        pass
    else:
        setting['z_sample_weights'] = float(setting['z_sample_weights'])

    setting['z_seed'] = int(setting['z_seed'])
    setting['z_subset_observations'] = int(setting['z_subset_observations'])

    # En las columnas `z_train_split` y `z_test_split` en los experimentos antiguos quedaron mal definidos. Por eso se debe cambiar aquellos en donde los valores no quedaron entre 0 y 1 sino entre 0 y 100.
    setting['z_test_split'] = float(setting['z_test_split']) / 100 if float(
        setting['z_test_split']) > 1 else float(
        setting['z_test_split'])
    setting['z_train_split'] = float(setting['z_train_split']) / 100 if float(
        setting['z_train_split']) > 1 else float(setting['z_train_split'])

    return setting
