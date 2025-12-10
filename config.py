from dc1.net_4 import Net as NET

# Uncomment the line below if you want to choose another architectures
# from dc1.experimental_nets import Net_3 as NET

# define the model above in the imports if you want to choose the architecture 6 (Arch. 2 in paper)
Net = NET

# Activate hyper parameter tuning, set to True if you want to tune the hyperparameters, False otherwise
hyptune = False

# set the number of epochs
nb_epochs = 20

# set the batch size 
batch_size = 25