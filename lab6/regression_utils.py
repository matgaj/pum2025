import numpy as np
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt


def plot_roc_curve(y, preds, y_train=None, y_pr_train=None, image_path=None):

    fpr, tpr, thresholds = roc_curve(y, preds)
    plt.plot([0,1], [0,1], linestyle='--', label='Chance level')
    plt.plot([0, 0, 1], [0, 1, 1], 'k--', label='Perfect model')
    plt.plot(fpr, tpr, marker='.', label='Evaluated model')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if y_train is not None:
        fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_pr_train)
        plt.plot(fpr_train, tpr_train, marker='.', label='Train model')
    plt.legend()
    if image_path is not None:
        plt.savefig(image_path, dpi = 300)   # zapis wykresu do pliku


def get_threshold(y_train, y_test, train_preds, test_preds):
    y = np.append(y_train, y_test)
    preds = np.append(train_preds, test_preds)
    fpr, tpr, thresholds = roc_curve(y, preds)
    gmeans = np.sqrt(tpr * (1-fpr)) #średnie geometryczne wyznaczone dla każdego punktu na krzywej ROC
    ix = np.argmax(gmeans) # indeks wartości maksymalnej
    print('Best Threshold=%.3f' % thresholds[ix])
    return thresholds[ix], (fpr[ix], tpr[ix]) # wartośc progu oraz współrzędne punktu na krzywej ROC