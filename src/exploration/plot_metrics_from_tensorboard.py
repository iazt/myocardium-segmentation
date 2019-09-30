import matplotlib.pyplot as plt
import pandas as pd
plt.style.use('seaborn-paper')

SMALL_SIZE = 15
MEDIUM_SIZE = 15
BIGGER_SIZE = 15

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

directory = r'C:\Users\Rudy\PycharmProjects\myocardium-segmentation\resources\metrics'
train_loss_path = directory + r'\run-.-tag-loss.csv'
val_loss_path = directory + r'\run-.-tag-val_loss.csv'
train_iou_path = directory + r'\run-.-tag-iou.csv'
val_iou_path = directory + r'\run-.-tag-val_iou.csv'

train_loss = pd.read_csv(train_loss_path)
val_loss = pd.read_csv(val_loss_path)
train_iou = pd.read_csv(train_iou_path)
val_iou = pd.read_csv(val_iou_path)

# Loss plot
plt.plot(train_loss['Step'], train_loss['Value'], label='Training')
plt.plot(val_loss['Step'], val_loss['Value'], label='Validation')
plt.grid()
plt.title('Loss evolution over epochs')
plt.xlabel('Epoch')
plt.ylabel('Jaccard loss')
plt.legend()
plt.show()

plt.figure()

# Loss plot
plt.plot(train_iou['Step'], train_iou['Value'], label='Training')
plt.plot(val_iou['Step'], val_iou['Value'], label='Validation')
plt.grid()
plt.title('IoU evolution over epochs')
plt.xlabel('Epoch')
plt.ylabel('IoU')
plt.legend()
plt.show()
