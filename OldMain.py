import torch
from models import Autoencoder, DeeperAutoencoder, WiderAutoencoder, DeeperWiderAutoencoder, NarrowerAutoencoder, ResNetAutoencoder
from HelperFunctions import Datasets
from Model_Helpers import test
 


# Load dataset
#train_loader, test_loader = load_data()
train_loader, test_loader = Datasets.load_np_data()

#Initialize the model
#model = Autoencoder()
#model = DeeperAutoencoder()
#model = WiderAutoencoder()
model = DeeperWiderAutoencoder()
#model = NarrowerAutoencoder()
#model = ResNetAutoencoder(channels=4)
model.cuda() # Move the model to the GPU

# Train the model
#train_model(model, train_loader, test_loader, save_path='Simple_autoencoder.pth')
#train_model(model, train_loader, test_loader, save_path='Deeper_autoencoder.pth')
#train_model(model, train_loader, test_loader, save_path='Wider_autoencoder.pth')
#train_model(model, train_loader, test_loader, save_path='DeeperWider_autoencoder.pth')
#train_model(model, train_loader, test_loader, save_path='Narrower_autoencoder.pth')
#train_model(model, train_loader, test_loader, save_path='Simple_autoencoder_with_smoothing.pth')
#train_model(model, train_loader, test_loader, save_path='ResNet_autoencoder.pth')

# Load the trained model
#model = Autoencoder()
#model = DeeperAutoencoder()
#model = WiderAutoencoder()
model = DeeperWiderAutoencoder()
#model = NarrowerAutoencoder()
#model = ResNetAutoencoder(channels=4)

#model.load_state_dict(torch.load('Simple_autoencoder.pth'))
#model.load_state_dict(torch.load('Deeper_autoencoder.pth'))
#model.load_state_dict(torch.load('Wider_autoencoder.pth'))
model.load_state_dict(torch.load('DeeperWider_autoencoder.pth'))
#model.load_state_dict(torch.load('Narrower_autoencoder.pth'))
#model.load_state_dict(torch.load('Simple_autoencoder_with_smoothing.pth'))
#model.load_state_dict(torch.load('ResNet_autoencoder.pth'))
model.eval()
model.cuda()

# Validate the model
#validate(train_loader, model)
test_dataset = Datasets.load_test()
test(test_dataset, model)