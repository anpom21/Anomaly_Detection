




import random
import time
#rom IPython.display import clear_output, display
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import sys, select
from src.Dataloader import Dataloader

dataset_path = 'Datasets/Dataset004'
dataset_path = 'Datasets/IRL_3_channel_dataset'
dataset = Dataloader(dataset_path)
train_loader, val_loader, test_loader = dataset.load_train_vali_test_dataloaders_with_n_images(n_images=3, trainSplit=0.8, BS=1)
test_loader = dataset.load_test_dataloader(3,1,"/Test_Small_defects") 


seed = 42
random.seed(seed)
# Prepare shuffled indices for test set
test_indices = list(range(len(test_loader.dataset)))
random.shuffle(test_indices)

human_preds = []
human_truths = []

global keypressed, key
keypressed = False
key = 0

def on_press(event):
    # print(f"Key pressed: {event.key}")
    global keypressed, key
    keypressed = True
    if event.key == '1':
        key = 1
    elif event.key == '0':
        key = 0
    else:
        print("Invalid key pressed. Please press 1 for anomaly (NOK) or 0 for normal (OK).")
    plt.close()

print("Press 1 for anomaly (NOK), 0 for normal (OK). You have 5 seconds per image.")
i = 0
for idx in test_indices:
    img, label = test_loader.dataset[idx]

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    plt.suptitle(f"Image {len(human_preds)}/{len(test_indices)}", fontsize=16)
    fig.canvas.mpl_connect('key_press_event', on_press)

    plt.subplot(1,3,1)
    plt.imshow(img.squeeze()[0].cpu().numpy(), cmap='gray')
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(img.squeeze()[1].cpu().numpy(), cmap='gray')
    plt.title("Is this image anomalous? 1:(NOK), 0:(OK)")
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.imshow(img.squeeze()[2].cpu().numpy(), cmap='gray')
    plt.axis('off')
    fig.canvas.manager.window.wm_geometry("+100+100")
    plt.show()


    pred = None
    print(f"img, no: {i}/{len(test_indices)}. Input (1=anomaly, 0=normal): ", end="", flush=True)


    while not keypressed:
        time.sleep(0.1)
        print('.', end='', flush=True)
    
    keypressed = False
    inp = key

    print(f"Input received: {inp}")

    human_preds.append(inp if isinstance(inp, int) else int(inp))
    human_truths.append(label if isinstance(label, int) else int(label))
    # clear_output(wait=True)
    i += 1

print("Human operator predictions:", human_preds)
print("Human operator truths:", human_truths)

# Confusion matrix
cm = confusion_matrix(human_truths, human_preds)
print("Confusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['OK', 'NOK'])
disp.plot()
plt.title('Human Operator Confusion Matrix')
plt.show()