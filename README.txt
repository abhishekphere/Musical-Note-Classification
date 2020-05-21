Each folder contains code and parameters.

There is a helper.py file with all the functions for loading, training and testing.

There is a separate folder with a waveform example of each class and each source
titled per_class_per_source_visualizations.

all the trained parameters have been stored in the respective folders.
they have been saved using the command 

torch.save(model.state_dict(), 'lstm_model.ckpt') <-- weights of the model

torch.load() and either the model or state options will work.

? python3 <program_name.py> is the only command required to run the code.

All code save visualizations without showing them. Please check folder where 
the code is stored after the run finishes for visualizations. 

All codes also save the parameters.
