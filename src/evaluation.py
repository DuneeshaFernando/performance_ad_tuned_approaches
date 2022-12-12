from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

def evaluation(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_pred)
    except:
        auc = 0
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print("Accuracy : ", round(accuracy, 2))
    print("Precision : ", round(precision, 2))
    print("Recall : ", round(recall, 2))
    print("AUC : ", round(auc, 2))
    print("F1 score : ", round(f1, 2))
    print("Confusion Matrix : \n", cm)