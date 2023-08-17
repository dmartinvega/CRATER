import pandas as pd
from tqdm import tqdm


def read_csvs_lightcurves_interpolated_and_classification():
    path_df_transient_lightcurves_interpolated = '/home/daniela/Documents/Academico/Maestria/astro_transient_class_css/data/raw/transient_lightcurves_interpolated_150_days.csv'
    df_transient_lightcurves_interpolated = pd.read_csv(path_df_transient_lightcurves_interpolated,
                                                        error_bad_lines=False)
    path_df_transient_classification = '/home/daniela/Documents/Academico/Maestria/astro_transient_class_css/data/raw/classification_transient_lightcurves_triggered_selected_13_classes.csv'
    df_transients_days_classification_and_t0 = pd.read_csv(path_df_transient_classification, error_bad_lines=False)
    return df_transient_lightcurves_interpolated, df_transients_days_classification_and_t0


def get_class_vector(is_pretransient_identified, df_transient_classification):
    df_transient_lightcurves_interpolated, df_transients_days_classification_and_t0 = read_csvs_lightcurves_interpolated_and_classification()

    # 2 nuevos dataframes vacíos
    # dataframe 1: crearle una columna para el id del evento y otra columna de la clasificación, filas del 0 al 149. Crearle 1 columna de clasificación: pre-transient y 13 clases

    if df_transient_classification is not None:
        df_transients_days_classification_and_t0 = df_transient_classification

    class_vector = pd.DataFrame(columns=['ID', 'MJD', 'Flux', 'Classification'])

    # recorrer cada evento
    for index, row in tqdm(df_transients_days_classification_and_t0.iterrows()):

        current_series = df_transient_lightcurves_interpolated.ID == row['TranID']
        df_transient_lightcurve_interpolated = df_transient_lightcurves_interpolated.loc[current_series]

        # Añadir subset al nuevo dataframe
        class_vector = class_vector.append(df_transient_lightcurve_interpolated[['ID', 'MJD', 'Flux']])

        if is_pretransient_identified:
            class_vector.loc[current_series, 'Classification'] = \
                class_vector['MJD'].apply(
                    lambda x: "Pre-transient" if x < row['t0'] else row['Classification'])
        else:
            class_vector.loc[current_series, 'Classification'] = row['Classification']

    # dataframe 1: ponerle el nombre del evento
    # extraer el t_0

    # por cada observación validar si el MJD es menor a t_0
    # si es menor, guardar el MJD en una columna y el número 0 (df2) en otra columna o la clasificación pre-transient
    # si es mayor o igual a t_0, poner el número 1 (df2) o la clasificación que le corresponde (df1)

    return class_vector


def get_class_matrix(is_pretransient_identified=True, filename='data/raw/classification_matrix.csv',
                     df_transient_classification=None):
    classification_vector = get_class_vector(is_pretransient_identified, df_transient_classification)
    classification_matrix = pd.get_dummies(classification_vector, columns=['Classification'])
    classification_matrix.to_csv(filename)

# get_class_matrix()
