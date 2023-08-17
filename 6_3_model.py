import os
from pickle import load, dump
import numpy as np
import mlflow.tensorflow
from keras.layers import GRU, Bidirectional, Dense, Dropout, BatchNormalization, TimeDistributed
from keras.models import Sequential
from keras.models import load_model
from tensorflow.keras import optimizers
from sklearn.preprocessing import MinMaxScaler
import mlflow

# plot_metrics = __import__('6_7_plot_metrics')


mlflow.tensorflow.autolog(every_n_iter=2)

# %cd "/content/drive/MyDrive/Academico/xx/" # colab o jupyter
os.chdir("/home/daniela/Documents/Academico/Maestria/astro_transient_class_css")
mlflow.set_tracking_uri("sqlite:///mlflow/tracking.db")
mlflow.set_registry_uri("sqlite:///mlflow/registry.db")
try:
    mlflow.create_experiment('astro_transient_class_css', "mlflow/")
except:
    print("Experiment already created")
mlflow.set_experiment("astro_transient_class_css")


def load_scale_and_train(setting):
    x_train, x_test, y_train, y_test = load_training_and_test_sets(setting["z_x_train_name"], setting["z_x_test_name"],
                                                                   setting["z_y_train_name"], setting["z_y_test_name"])

    x_train_scaled, x_test_scaled = scale_data(x_train, x_test, setting["z_x_train_scaled"], setting["z_x_test_scaled"])

    return run_experiment(setting, x_train_scaled, x_test_scaled, y_train, y_test)


def run_experiment(setting, x_train, x_test, y_train, y_test, retrain=True):
    with mlflow.start_run(run_name=setting["z_run_name"]):
        if not retrain and os.path.isfile("models/" + setting["z_run_name"]):
            model = load_model("models/" + setting["z_run_name"])
        else:
            model = train_model(setting, x_train, x_test, y_train, y_test)

            # estimated_labels = model.predict()
            # metrics = calculate_metrics(real_labels, estimated_labels)

            mlflow.log_params(setting)
            # mlflow.log_metrics(metrics)

        return model


def load_training_and_test_sets(x_train_name, x_test_name, y_train_name, y_test_name):
    x_train = load(open('data/preprocessed/' + x_train_name + '.pkl', 'rb'))
    x_test = load(open('data/preprocessed/' + x_test_name + '.pkl', 'rb'))
    y_train = load(open('data/preprocessed/' + y_train_name + '.pkl', 'rb'))
    y_test = load(open('data/preprocessed/' + y_test_name + '.pkl', 'rb'))
    return x_train, x_test, y_train, y_test


def scale_data(x_train, x_test, x_train_scaled_filename, x_test_scaled_filename):
    minimum = x_train.min()
    maximum = x_train.max()
    x_train_scaled = (x_train - minimum) / (maximum - minimum)
    x_test_scaled = (x_test - minimum) / (maximum - minimum)
    dump(x_train_scaled, open('data/preprocessed/' + x_train_scaled_filename + '.pkl', 'wb'))
    dump(x_test_scaled, open('data/preprocessed/' + x_test_scaled_filename + '.pkl', 'wb'))
    return x_train_scaled, x_test_scaled


def train_model(setting, x_train, x_test, y_train, y_test):
    num_classes = y_test.shape[1]
    y_train = np.transpose(y_train, (0, 2, 1))
    y_test = np.transpose(y_test, (0, 2, 1))

    model = Sequential()

    if setting["z_bidirectional"]:
        model.add(Bidirectional(GRU(setting["z_gru_size_1"], return_sequences=setting["z_return_sequences"])))
    else:
        model.add(GRU(setting["z_gru_size_1"], return_sequences=setting["z_return_sequences"]))

    model.add(Dropout(setting["z_dropout_rate"], seed=setting["z_seed"]))
    model.add(BatchNormalization())

    if setting["z_bidirectional"]:
        model.add(Bidirectional(GRU(setting["z_gru_size_1"], return_sequences=setting["z_return_sequences"])))
    else:
        model.add(GRU(setting["z_gru_size_1"], return_sequences=setting["z_return_sequences"]))

    model.add(Dropout(setting["z_dropout_rate"], seed=setting["z_seed"]))
    model.add(BatchNormalization())

    if setting["z_bidirectional"]:
        model.add(Bidirectional(GRU(setting["z_gru_size_2"], return_sequences=setting["z_return_sequences"])))
    else:
        model.add(GRU(setting["z_gru_size_2"], return_sequences=setting["z_return_sequences"]))

    model.add(Dropout(setting["z_dropout_rate"], seed=setting["z_seed"]))
    model.add(BatchNormalization())
    model.add(Dropout(setting["z_dropout_rate"], seed=setting["z_seed"]))

    model.add(TimeDistributed(Dense(500, activation=setting["z_activation"])))
    model.add(TimeDistributed(Dense(num_classes, activation=setting["z_activation"])))

    if setting["z_learning_rate"] is not None:
        opt = optimizers.Adam(learning_rate=float(setting["z_learning_rate"]))
    elif setting["z_base_lr"] is not None and setting["z_decay_steps"] is not None and setting[
        "z_end_lr"] is not None and setting["z_power"] is not None:
        polinomial_decay = optimizers.schedules.PolynomialDecay(setting["z_base_lr"], setting["z_decay_steps"],
                                                                setting["z_end_lr"],
                                                                power=setting["z_power"])
        opt = optimizers.Adam(learning_rate=polinomial_decay)  # optimizer
    model.compile(loss=setting["z_loss"], optimizer=opt, metrics=['accuracy'])

    model.build(input_shape=x_test.shape)

    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=setting["z_epochs"],
              batch_size=setting["z_batch_size"], verbose=2,
              sample_weight=setting["z_sample_weights"])

    # Guardar archivos
    if "rapid" in setting["z_run_name"]:
        mlflow.log_artifact("data/preprocessed/y_test_reshaped.pkl")
        mlflow.log_artifact("data/preprocessed/x_train_scaled.pkl")
        mlflow.log_artifact("data/preprocessed/x_test_scaled.pkl")
        mlflow.log_artifact("data/preprocessed/x_test_reshaped.pkl")
        mlflow.log_artifact("data/preprocessed/y_train_reshaped.pkl")
        mlflow.log_artifact("data/preprocessed/x_train_reshaped.pkl")
        mlflow.log_artifact("data/preprocessed/y_train.pkl")
        mlflow.log_artifact("data/preprocessed/y_test.pkl")
        mlflow.log_artifact("data/preprocessed/x_train.pkl")
        mlflow.log_artifact("data/preprocessed/x_test.pkl")
        mlflow.log_artifact("data/preprocessed/tran_id_classification_label_encoder.pkl")
    elif "german" in setting["z_run_name"]:
        mlflow.log_artifact("data/preprocessed/y_test_reshaped.pkl")
        mlflow.log_artifact("data/preprocessed/x_train_scaled.pkl")
        mlflow.log_artifact("data/preprocessed/x_test_scaled.pkl")
        mlflow.log_artifact("data/preprocessed/x_test_reshaped.pkl")
        mlflow.log_artifact("data/preprocessed/y_train_reshaped.pkl")
        mlflow.log_artifact("data/preprocessed/x_train_reshaped.pkl")
        mlflow.log_artifact("data/preprocessed/tran_id_classification_label_encoder.pkl")

    with open("metrics/model_summary.txt", 'w') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\n'))
    mlflow.log_artifact("metrics/model_summary.txt")
    model.save("models/" + setting["z_run_name"])

    return model


# def calculate_metrics(real_labels, estimated_labels):


def main(setting):
    return load_scale_and_train(setting)

# if __name__ == '__main__':
#     main()

# main()
