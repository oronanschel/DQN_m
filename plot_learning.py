import matplotlib.pyplot as plt
import csv, pdb, sys
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

def plot(filename, pdf_loc="training.pdf", csv_loc="training_progress.csv"):
  filename = "models/2016-01-15_14-36-05" if filename is None else filename
  open_loc = "%s/%s" % (filename, csv_loc)
  save_loc = "%s/%s" % (filename, pdf_loc)
  with open(open_loc) as csvfile:
    reader = csv.DictReader(csvfile)
    epochs = []
    scores = []
    q_vals = []
    for row in reader:
      epochs.append(row['epoch'])
      scores.append(float(row['mean_score']))
      q_vals.append(float(row['mean_q_val']))
  smooth_scores = [np.mean(scores[max(0,i-10):i+1]) for i in range(len(scores))]
  fig, ax1 = plt.subplots()
  ax1.plot(epochs, scores, "r", label="testing score")
  ax1.plot(epochs, smooth_scores, "g", label="10 moving avg")
  spread = (max(scores) - min(scores))*1.1
  if spread > 0:
    ax1.set_ylim([max(scores)-spread,min(scores)+spread])

  box = ax1.get_position()
  ax1.set_position([box.x0, box.y0, box.width * 0.75, box.height])
  ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

  ax2 = ax1.twinx()
  ax2.plot(epochs, q_vals, 'b', label="avg q vals")
  spread = (max(q_vals) - min(q_vals))*1.1
  ax2.set_ylim([max(q_vals)-spread,min(q_vals)+spread])
  box = ax2.get_position()
  ax2.set_position([box.x0, box.y0, box.width * 0.75, box.height])
  ax2.legend(loc='center left', bbox_to_anchor=(1, 0.3))
  #plt.show()
  pp = PdfPages(save_loc)
  pp.savefig(fig)
  pp.close()
  plt.close()

if __name__ == "__main__":
  plot(sys.argv[1] if len(sys.argv) > 1 else None)