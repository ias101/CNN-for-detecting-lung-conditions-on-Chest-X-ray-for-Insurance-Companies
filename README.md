# CNN for detecting lung conditions on chest X-ray for insurance companies

This work is done by students from TU/e university

## Code structure
There are ten `.py` files in total, each containing a different part of the code. 
Feel free to create new files to explore the data or experiment with other ideas.

- To download the data: run the `ImageDataset.py` file. The script will create a directory `/data/` and download the training and test data with corresponding labels to this directory.
- To download the preprocessed data after: run the `data_augmentation.ipynb` file, then it is going to create one more data folder nest to the previously created one

- To run the whole training/evaluation pipeline: run `pipeline_main.py`. This script is prepared to do the followings:
    - Load your train and test data (Make sure its downloaded beforehand!)
    - Initializes the neural network as defined in the `Net.py` file. (if you want to choose another model just change the imports at `config.py`) Namely you can choose one of the 8th architectures `experimental_nets.py` there is 8 different architectures.
    - Define number of training epochs and batch size in the terminal
    - Train the neural network and perform evaluation on test set at the end of each epoch. 
    - Provide plots about the training losses both during training in the command line and as a png (saved in the `/artifacts/` subdirectory, but additional statistics is going to be printed in the terminal)
    - Finally, save your trained model's weights in the `/model_weights/` subdirectory so that you can reload them later.


## GitHub setup instructions
1. Click the green *<> Code* button at the upper right corner of the repository.
2. Make sure that the tab *Local* is selected and click *Download ZIP*.
3. Go to the GitHub homepage and create a new repository.
4. Press *uploading an existing file* and upload the extracted files from JBG040-Group15-main.zip to your repository. Note that for the initial commit you should commit directly to the main branch
5. Open PyCharm and make sure that your GitHub account is linked.*
6. In the welcome screen of PyCharm, click *Get from VCs > GitHub* and select your repository and click on clone.
(Alternatively you can use VS Code, in this case you just open the downloaded repository as a new folder)
7. After the repository is cloned, you can now create a virtual environment using the requirements.txt.

## Environment setup instructions
We recommend to set up a virtual Python environment to install the package and its dependencies. To install the package, we recommend to execute `pip install -r requirements.txt.` in the command line. This will install it in editable mode, meaning there is no need to reinstall after making changes. If you are using PyCharm, it should offer you the option to create a virtual environment from the requirements file on startup. Note that also in this case, it will still be necessary to run the pip command described above.

## Mypy
The code is created with support for full typehints. This enables the use of a powerful tool called `mypy`. Code with typehinting can be statically checked using this tool. It is recommended to use this tool as it can increase confidence in the correctness of the code before testing it. Note that usage of this tool and typehints in general is entirely up to the students and not enforced in any way. To execute the tool, simply run `mypy .`. For more information see https://mypy.readthedocs.io/en/latest/faq.html

## Argparse
Argparse functionality is included in the `pipeline_main.py` file. This means the file can be run from the command line while passing arguments to the main function. Right now, there are arguments included for the number of epochs (nb_epochs), batch size (batch_size), and whether to create balanced batches (balanced_batches). You are free to add or remove arguments as you see fit.

To make use of this functionality, first open the command prompt and change to the directory containing the `pipeline_main.py` file.
For example, if you're main file is in C:\Data-Challenge-1-template-main\dc1\, 
type `cd C:\Data-Challenge-1-template-main\dc1\` into the command prompt and press enter.

Then, `pipeline_main.py` can be run by, for example, typing `python pipeline_main.py --nb_epochs 10 --batch_size 25`.
This would run the script with 10 epochs, a batch size of 25, and balanced batches, which is also the current default.
If you would want to run the script with 20 epochs, a batch size of 5, and batches that are not balanced, 
you would type `pipeline_main.py --nb_epochs 20 --batch_size 5 --no-balanced_batches`.

But we recommend to set this changes by default in `config.py` before running the `pipeline_main.py` file.

## Structure of files

1. `pipeline_main` is the main file where the general algorithms are defined and executed
2. `config.py` is the configuration file where you can set the configurations for the code (descriptions is in the file)
3. `image_dataset.py` is the file where you can download the datasets
4. `Data_Augmentation.ipynb` is the file which you run to preprocess the images and save them in the data directory.
