import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pickle import dump
import importlib


def load_data():
    X = pd.read_csv("./data/raw/Datos german/means.txt", sep=" ", header=None)
    y = pd.read_csv("./data/raw/Datos german/labs.txt", header=None)
    return X, y


def select_from_classes(classes, X, y):
    y = y[y.isin(classes)].dropna()
    X = X.loc[y.index, :]
    return X, y


def subset_observations(X, subset_number):
    length_transient_lightcurves_columns = len(X.columns)

    if subset_number < length_transient_lightcurves_columns:
        df_transient_lightcurves_classification_without_label_limit_cut_lower = int(
            (length_transient_lightcurves_columns / 2) - (subset_number / 2) - 1)
        df_transient_lightcurves_classification_without_label_limit_cut_upper = int(
            (length_transient_lightcurves_columns / 2) + (subset_number / 2) - 1)
        X = X.iloc[:,
            df_transient_lightcurves_classification_without_label_limit_cut_lower:df_transient_lightcurves_classification_without_label_limit_cut_upper]

    return X


def encode_classification(X, y):
    df_transient_lightcurves_classification = X.copy()
    df_transient_lightcurves_classification["Classification"] = y

    train_test_split_css = importlib.reload(__import__('6_1_train_test_split'))
    df_transient_lightcurves_classification_label_encoder = train_test_split_css.save_label_encoder(
        df_transient_lightcurves_classification)

    class_label_encoded = df_transient_lightcurves_classification_label_encoder['ClassLabelEncoded']
    class_label_encoded = pd.get_dummies(class_label_encoded)

    return class_label_encoded


def undersample_training_set(x_train_imbalanced, y_train_imbalanced, undersampling_mode):
    x_balanced = x_train_imbalanced
    y_balanced = y_train_imbalanced
    if undersampling_mode == 'equal':
        min_class_representation = y_train_imbalanced.value_counts().min()
        unique_serie = y_train_imbalanced.value_counts().index.to_frame()
        x_balanced = []
        y_balanced = pd.DataFrame()
        for row in unique_serie.iterrows():
            indice = pd.merge(y_train_imbalanced.reset_index(drop=True).reset_index(), row[1].to_frame().T, how="inner")["index"]
            lista = np.random.choice(indice, size=min_class_representation)
            y_balanced = pd.concat([y_balanced, y_train_imbalanced.iloc[lista, :]])
            x_balanced.append(x_train_imbalanced[lista, :, :])
        x_balanced = np.concatenate(x_balanced)
    return x_balanced, y_balanced


def save_train_and_test_data(x_train, x_test, y_train_reshaped, y_test_reshaped, x_train_filename,
                             x_test_filename, y_train_filename, y_test_filename):
    dump(x_train, open('data/preprocessed/' + x_train_filename + '.pkl', 'wb'))
    dump(x_test, open('data/preprocessed/' + x_test_filename + '.pkl', 'wb'))
    dump(y_train_reshaped, open('data/preprocessed/' + y_train_filename + '.pkl', 'wb'))
    dump(y_test_reshaped, open('data/preprocessed/' + y_test_filename + '.pkl', 'wb'))


def model_training_with_german_input_data(setting):
    df_transient_lightcurves_classification_without_label, y = load_data()

    if setting["z_classes"] is not None:
        df_transient_lightcurves_classification_without_label, y = select_from_classes(setting["z_classes"],
                                                                                       df_transient_lightcurves_classification_without_label,
                                                                                       y)
    df_transient_lightcurves_classification_without_label = subset_observations(
        df_transient_lightcurves_classification_without_label, setting["z_subset_observations"])

    class_label_encoded = encode_classification(df_transient_lightcurves_classification_without_label, y)

    df_transient_lightcurves_classification_without_label_reshaped = df_transient_lightcurves_classification_without_label.to_numpy()[
                                                                     :, :, np.newaxis]

    x_train, x_test, y_train, y_test = train_test_split(
        df_transient_lightcurves_classification_without_label_reshaped, class_label_encoded,
        train_size=setting["z_train_split"],
        test_size=setting["z_test_split"], random_state=42)

    if setting["z_undersampling_mode"] is not None:
        x_train, y_train = undersample_training_set(x_train, y_train, setting["z_undersampling_mode"])

    y_train_reshaped = np.tile(y_train.to_numpy()[:, :, np.newaxis], reps=setting["z_subset_observations"])
    y_test_reshaped = np.tile(y_test.to_numpy()[:, :, np.newaxis], reps=setting["z_subset_observations"])

    save_train_and_test_data(x_train, x_test, y_train_reshaped, y_test_reshaped,
                             setting["z_x_train_name"], setting["z_x_test_name"], setting["z_y_train_name"],
                             setting["z_y_test_name"])

    run_model = importlib.reload(__import__('6_3_model'))
    run_model.main(setting)
