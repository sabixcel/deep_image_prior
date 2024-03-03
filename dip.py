import numpy as np
#core Python library for N-D arrays, scientific programming
import matplotlib.pyplot as plt #plotting library
import cv2
#OpenCV library (for computer vision problems)
import torch, torch.nn as nn
#provides deep learning components
from skimage.transform import resize #scikit-image processing library (aka skimage): set of algorithms
from skimage.data import chelsea
#the image used (image of a cat)

torch.manual_seed(0); #fix the random numbers used to get exact same sequence of random numbers)
plt.rcParams["font.size"] = "14"; plt.rcParams['toolbar'] = 'None' #adjusting the parameters of matplotlib, font size=14 and remove the tool bar plt.ion() #switch on interactive mode
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); print(device) #if there is no gpu, it will be used the cpu
nxd = 128 #number of pixels in x dimension
chelseaimage - chelsea() #cat image
true_object_np = 100.0*resize(chelseaimage[10:299,110:399,2], (nxd,nxd), anti_aliasing=False) #the true object is used here only to compare the results (because in reality we do not have it)
fig1, axs1 = plt.subplots(2,3, figsize=(20,12)) #create a figure and a set of axis, 2 rows and 3 col => 6 subplots

#cv2.waitKey(10000)

plt.tight_layout(); fig1.canvas.manager.window.move(0,0) #tight layout to maximize the images and plots and move the window to the top right corner optional
axs1[0,2].imshow(true_object_np, cmap='Greys_r'); #take a look at the true object in grayscale color map
axs1[0,2].set_axis_off(); #switch off the numbered axes

# Now set up a CNN class #=========
class CNN (nn.Module): #the class CNN is inherit from the torch.nn.Module, which got all the convolutional layers and ReLU's we need to define the CNN
    def __init__(self, num_channels): #when the class is called there is an initialization method that will be called
        super(CNN, self).__init__()
        self.CNN = nn.Sequential(
            nn.Conv2d(1, num_channels, 3, padding=1), nn.PReLU, #a 2d conv layer, there is 1 single image at the input, num_channels - the kernels, the image will be on size 3x3, #then pad the image with 1 pixel all around the perimeter, so that we can evolve with a 3x3 kernel; there also be a bias, and through a non-linearity a parametric PRELU nn.Conv2d(num_channels, num_channels, 3, padding=1), nn.PReLU, #will take the number of channels from the first layer output as input
            nn.Conv2d(num_channels, num_channels, 3, padding=1), nn.PReLU,
            nn.Conv2d(num_channels, num_channels, 3, padding=1), nn.PReLU,
            nn.Conv2d(num_channels, num_channels, 3, padding=1), nn.PReLU,
            nn.Conv2d(num_channels, num_channels, 3, padding=1), nn.PReLU,
            nn.Conv2d(num_channels, num_channels, 3, padding=1), nn.PReLU,
            nn.Conv2d(num_channels, num_channels, 3, padding=1), nn.PReLU,
            nn.Conv2d(num_channels, num_channels, 3, padding=1), nn.PReLU,
            nn.Conv2d(num_channels, num_channels, 3, padding=1), nn.PReLU,
            nn.Conv2d(num_channels, num_channels, 3, padding=1), nn.PReLU,
            nn.Conv2d(num_channels, num_channels, 3, padding=1), nn.PReLU,
            nn.Conv2d(num_channels, num_channels, 3, padding=1), nn.PReLU,
            nn.Conv2d(num_channels, num_channels, 3, padding=1), nn.PReLU,
            nn.Conv2d(num_channels, num_channels, 3, padding=1), nn.PReLU,
            nn.Conv2d(num_channels, num_channels, 3, padding=1), nn.PReLU,
            nn.Conv2d(num_channels, num_channels, 3, padding=1), nn.PReLU,
            nn.Conv2d(num_channels, num_channels, 3, padding=1), nn.PReLU,
            nn.Conv2d(num_channels, num_channels, 3, padding=1), nn.PReLU,
            nn.Conv2d(num_channels, num_channels, 3, padding=1), nn.PReLU,
            nn.Conv2d(num_channels, num_channels, 3, padding=1), nn.PReLU,
            nn.Conv2d(num_channels, 1, 3, padding=1), nn.PReLU, #we have num_channels features maps coming in to the conv layer, and then 1 kernel with that many channels to it at output
        )
    def forward(self, x): return torch.squeeze(self.CNN (x.unsqueeze(0).unsqueeze (0)))

#=========
# MORE EFFICIENT OPTION FOR CNN CLASS
class CNN_configurable (nn.Module):
    def __init__(self, n_lay, n_chan, ksize):
        super(CNN_configurable, self).__init__()
        pd = int(ksize/2)
        layers = [nn.Conv2d(1, n_chan, ksize, padding=pd), nn.PReLU(),]
        for _ in range(n_lay):
            layers.append(nn.Conv2d(n_chan, n_chan, ksize, padding=pd)); layers.append(nn.PReLU()) layers.append(nn.Conv2d(n_chan, 1, ksize, padding=pd)); layers.append(nn.PReLU())
        self.deep_net = nn.Sequential(*layers)
    def forward(self, x):
        return torch.squeeze(self.deep_net(x.unsqueeze (0).unsqueeze (0)))

cnn = CNN_configurable(32, nxd, 3).to(device) #the cnn is instantiate as an object that have 32 layers, 128 channels, 3x3 kernels

input_image = torch.rand(nxd, nxd).to(device) #the input image is a random array 128x128 (actually a torch tensor)
#torch tensor = n-dimensional arrays that allows gradients to be propagated through them
#TORCH TO NUMPY CONVERTORS
def torch_to_np(torch_array): return np.squeeze(torch_array.detach().cpu().numpy())
def np_to_torch(np_array): return torch.from_numpy(np_array).float()

true_object_torch = np_to_torch (true_object_np).to(device)

#NOISY EXAMPLE
measured_data = torch.poisson (true_object_torch) #the measured image data is the true object torch tensor; torch.poisson make a noisy version of the image #GAP EXAMPLE
mask_image = torch.ones_like(measured_data)
mask_image[int(0.65*nxd): int(0.85*nxd), int(0.65*nxd):int(0.85*nxd)] = 0

mask_image[int (0.15*nxd): int(0.25*nxd), int(0.15*nxd): int(0.35*nxd)] = 0

measured_data = measured_data * mask_image

axs1[0,2].imshow(torch_to_np(true_object_torch), cmap='Greys_r'); axs1[0,2].set_title('TRUE'); axs1[0,2].set_axis_off();
axs1[0,1].imshow(torch_to_np(measured_data), cmap='Greys_r'); axs1[0,1].set_title('DATA'); axs1[0,1].set_axis_off();
axs1[1,0].imshow(torch_to_np(input_image), cmap='Greys_r'); axs1[1,0].set_title('z image %d x %d' % (nxd, nxd)); axs1[1,0].set_axis_off();
#the input image is the one we are going to fix and put into the CNN
cv2.waitKey(10000)


optimiser = torch.optim.Adam(cnn.parameters(), lr=1e-4) #for training the network we need an optimizer (we use Adam optimizer, a more sophisticated gradient descent method), #then we pass to it the object cnn, we specify the trainable parameters of the CNN and we provide the learning rate lr train_loss
list(); nrmse_list = list(); best_nrmse = 10e9 #we track the loss function, we use a normalized root mean square error
def nrmse_fn(recon, reference):
    n = (reference-recon)**2; den = reference**2
    return 100.0 * torch.mean(n)**0.5/ torch.mean(den)**0.5

for ep in range(1000000 +1): #training loop - we iteratively optimize the parameters of the cnn, over 1000000 iterations/epochs optimiser.zero_grad() #set the gradients to zero
    output_image = cnn(input_image) #we put the input_image into the cnn to generate an output
    loss = nrmse_fn(output_image*mask_image, measured_data*mask_image) #train on masked data - compare the measured_data with the output of the cnn, but for the masked region #for the loss function to take place only for the non-zero values in the image
    train_loss.append(loss.item())
    loss.backward() #find the gradients optimiser.step() #does the update
    nrmse = nrmse_fn(output_image, true_object_torch) #evaluate error wrt true image over all nrmse_list.append(nrmse.item())

    if nrmse < best_nrmse or ep == 0:
        best_recon = output_image; best_nrmse = nrmse; best_ep = ep
        axs1[1,2].cla(); axs1[1,2].imshow(torch_to_np(best_recon), cmap='Greys_r') axs1[1,2].set_title('Best Recon %d, NRMSE = %.2f%%' % (best_ep, best_nrmse)) axs1[1,2].set_axis_off();

    if ep % 2 == 0:
        axs1[1,1].cla(); axs1[1,1].imshow(torch_to_np(output_image), cmap='Greys_r') axs1[1,1].set_title('Recon %d, NRMSE = %.2f%%'% (ep, nrmse));
        axs1[1,1].set_axis_off();
        axs1[0,0].cla(); axs1[0,0].plot(train_loss [-200:-1]);
        axs1[0,0].plot(nrmse_list[-200:-1])
        axs1[0,0].set_title('NRMSE (%%), epoch %d % ep);
        axs1[0,0].set_axis_off();
        cv2.waitKey(1) #allow time for update
