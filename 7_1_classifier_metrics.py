import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from pickle import load

model_predict = __import__('6_4_model_predict')


# try:
#     import imageio
# except ImportError:
#     print("Warning: You will need to install 'matplotlib' and 'imageio' if you want to plot the "
#           "classification performance metrics.")


def main_with_model(y_test_path, model, x_test_filename):
    y_true = load(open(y_test_path, 'rb'))
    x_test_scaled = load(open('data/preprocessed/' + x_test_filename + '.pkl', 'rb'))

    y_pred = model.predict(x_test_scaled)

    get_confusion_matrix(y_true, y_pred)
    get_average_accuracy(y_true, y_pred)


def confusion_matrix_and_accuracy_every_5_days(y_test_path, model_directory, logged_model, x_test_filename,
                                               y_test_predicted_name):
    y_true = load(open(y_test_path, 'rb'))
    y_pred = predict_from_model(model_directory, logged_model, x_test_filename, y_test_predicted_name)

    get_confusion_matrix(y_true, y_pred)
    get_average_accuracy(y_true, y_pred)


def predict_from_model(run_id, model_directory, logged_model, x_test_filename, y_test_predicted_name):
    return model_predict.predict_or_load_model(run_id, model_directory, logged_model, x_test_filename,
                                               y_test_predicted_name)


from sklearn.metrics import accuracy_score


def get_confusion_matrix(y_true, y_pred_probabilities):
    # retornar el label de la clase en la que se encuentra cada evento con mayor probabilidad
    y_true_unique_label = np.argmax(y_true, axis=1)
    y_pred_unique_label = np.argmax(y_pred_probabilities, axis=2)

    # no tiene mucho sentido crear matrices de confusión para los primeros días del evento porque todos son
    # pre-transient. Tampoco se crea una matriz de confusión por día, porque serían demasiadas matrices y puede ser
    # innecesario

    for i in range(34, 150, 5):
        # for i in range(0, y_true_unique_label.shape[1]):
        print('{} days since trigger'.format(i - 69))

        print("y_true")
        print(y_true_unique_label[:, i])
        y_true_daily = y_true_unique_label[:, i]

        print("y_pred")
        print(y_pred_unique_label[:, i])
        y_pred_daily = y_pred_unique_label[:, i]
        print(confusion_matrix(y_true_daily, y_pred_daily))
        print("accuracy_score")
        print(accuracy_score(y_true_daily, y_pred_daily))
        print("--------\n")


def get_average_accuracy(y_true, y_pred):
    # print(accuracy_score(y_true, y_pred))
    pass


# confusion_matrix_and_accuracy_every_5_days('data/preprocessed/y_test_sub_reshaped.pkl',
#      "/home/daniela/Documents/Academico/Maestria/astro_transient_class_css",
#      'runs:/8bd15285cc2a48a09629473fb352d58e/model', 'x_test_sub_reshaped', 'y_test_sub_reshaped_predicted')

# confusion_matrix_and_accuracy_every_5_days('data/preprocessed/y_test_sub_reshaped.pkl',
#      "/home/daniela/Documents/Academico/Maestria/astro_transient_class_css",
#      'runs:/6d67c8fc07cd466da141fce02890f639/model', 'x_test_sub_scaled', 'y_test_sub_reshaped_predicted')

# confusion_matrix_and_accuracy_every_5_days('data/preprocessed/y_test_reshaped.pkl',
#      "/home/daniela/Documents/Academico/Maestria/astro_transient_class_css",
#      'runs:/6cd98870fa072cba550a12bec872bea2ac45a3d9/model', 'x_test_scaled', 'y_test_reshaped_predicted')


def predict_and_get_metrics(run_id, y_test_path, model_directory, logged_model, x_test_filename, y_test_predicted_name):
    y_true_unique_label, y_pred_unique_label = get_true_and_predicted_values(run_id, y_test_path, model_directory, logged_model,
                                                                             x_test_filename, y_test_predicted_name)
    return get_metrics(y_true_unique_label, y_pred_unique_label)


def get_true_and_predicted_values(run_id, y_test_path, model_directory, logged_model, x_test_filename, y_test_predicted_name):
    y_true = load(open(y_test_path, 'rb'))
    y_pred_probabilities = predict_from_model(run_id, model_directory, logged_model, x_test_filename, y_test_predicted_name)

    # retornar el label de la clase en la que se encuentra cada evento con mayor probabilidad
    y_true_unique_label = np.argmax(y_true, axis=1)
    y_pred_unique_label = np.argmax(y_pred_probabilities, axis=2)

    return y_true_unique_label, y_pred_unique_label


def get_metrics(y_true_unique_label, y_pred_unique_label):
    precision = []
    recall = []
    f1 = []

    for i in range(0, y_true_unique_label.shape[1]):
        precision.append(
            precision_score(y_true_unique_label[:, i], y_pred_unique_label[:, i], average='weighted'))
        recall.append(
            recall_score(y_true_unique_label[:, i], y_pred_unique_label[:, i], average='weighted'))
        f1.append(
            f1_score(y_true_unique_label[:, i], y_pred_unique_label[:, i], average='weighted'))

    return precision, recall, f1, None
