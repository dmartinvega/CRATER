from pickle import dump, load

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def read_csv_classification_lightcurves():
    path_df_transient_classification = '/home/daniela/Documents/Academico/Maestria/astro_transient_class_css/data/raw/classification_transient_lightcurves_triggered_selected_13_classes.csv'
    df_transient_lightcurves_classification = pd.read_csv(path_df_transient_classification)
    return df_transient_lightcurves_classification


def create_label_encoder(df_transient_lightcurves_classification):
    classes = df_transient_lightcurves_classification['Classification']
    le = preprocessing.LabelEncoder()
    le.fit(classes)
    df_transient_lightcurves_classification['ClassLabelEncoded'] = le.transform(classes)
    return le


def save_label_encoder(df_transient_lightcurves_classification):
    le = create_label_encoder(df_transient_lightcurves_classification)
    with open('data/preprocessed/tran_id_classification_label_encoder.pkl', 'wb') as descriptor:
        dump(le, descriptor)
    return df_transient_lightcurves_classification


def prepare_and_save_sets(tran_ids, class_label_encoded, train_size):
    x_train, x_test, y_train, y_test = train_test_split(tran_ids, class_label_encoded, train_size=train_size,
                                                        random_state=42)
    dump(x_train, open('data/preprocessed/x_train.pkl', 'wb'))
    dump(x_test, open('data/preprocessed/x_test.pkl', 'wb'))
    dump(y_train, open('data/preprocessed/y_train.pkl', 'wb'))
    dump(y_test, open('data/preprocessed/y_test.pkl', 'wb'))


def open_training_and_test_sets(x_train_name, x_test_name, y_train_name=None, y_test_name=None):
    # tran_id_classification_label_encoder = load(open('data/preprocessed/tran_id_classification_label_encoder.pkl', 'rb'))
    x_train = load(open('data/preprocessed/' + x_train_name + '.pkl', 'rb'))
    x_test = load(open('data/preprocessed/' + x_test_name + '.pkl', 'rb'))

    if y_train_name is not None or y_test_name is not None:
        y_train = load(open('data/preprocessed/' + y_train_name + '.pkl', 'rb'))
        y_test = load(open('data/preprocessed/' + y_test_name + '.pkl', 'rb'))
        return x_train, x_test, y_train, y_test
    else:
        return x_train, x_test


def inverse_transform_labels(df_transient_lightcurves_classification):
    le = create_label_encoder(df_transient_lightcurves_classification)
    return le.inverse_transform(list(range(13)))


if __name__ == "__main__":
    df_transient_lightcurves_classification = read_csv_classification_lightcurves()
    df_transient_lightcurves_classification_label_encoder = save_label_encoder(df_transient_lightcurves_classification)
    train_size = 0.7
    prepare_and_save_sets(df_transient_lightcurves_classification_label_encoder['TranID'],
                          df_transient_lightcurves_classification_label_encoder['ClassLabelEncoded'], train_size)
