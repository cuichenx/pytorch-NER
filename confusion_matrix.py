from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


run_name = "temp_grpdlid2_lstm_Clf_run1"
print("run_name is", run_name)


pred_file_path = f"evaluation/{run_name}/pred.txt"

with open(pred_file_path) as f:
    txt = f.read().split('\n')

y_test = []
y_pred = []
for line in txt:
    if line:
        line_split = line.split(' ')
        if len(line_split) == 3:
            word, gt, pred = line_split
            y_pred.append(pred)
            y_test.append(gt)
        else:
            print("this line is weird:", line)


ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title(run_name)
plt.savefig(f"{run_name}_confusion.png")
print("figure saved!")