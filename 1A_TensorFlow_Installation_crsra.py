#!/usr/bin/env python
# coding: utf-8

# # TensorFlow Installation

# A walkthrough to install TensorFlow for Python 3.6. If you already have TensorFlow on your machine and feel confident in your installation, you can go ahead and skip this notebook.
# 
# For the purposes of this tutorial, I'm recommending using Anaconda as your Python distribution and for managing your environments. If you'd rather use some other distribution or version of Python and feel confident in your abilities, you may also choose to just follow the instructions here: https://www.tensorflow.org/install/.
# 
# This tutorial is broken up by operating system (OS). Pick the instructions for the OS that you plan to use, and then check your install:

# ### Contents
# 1. Ubuntu
# 2. Windows
# 3. Mac OS X
# 4. Verify TensorFlow Installation

# ## Ubuntu

# 1. Download and install Anaconda
# 
#     1.1 Go [here](https://www.continuum.io/downloads#linux) and click on the Python 3.6 installer matching your system's architecture (likely 64-bit X86).
#     
#     1.2 Navigate to your downloads directory in a terminal and run the installer:
#     
#     ```Shell
#     cd ~/Downloads/
#     bash Anaconda3-5.2.0-Linux-x86_64.sh 
#     ```
# 
# 2. In a terminal, create a conda environment for TensorFlow, with Python 3.6:
# 
#     ```Shell
#     conda create -n tensorflow python=3.6
#     ```
#     
# 3. Activate your new environment:
# 
#     ```Shell
#     source activate tensorflow
#     ```
#     
#     You will see your prompt prepend `(tensorflow)` to indicate that you are in your `tensorflow` environment.
# 
# 4. Install some supporting dependencies
#     ```Shell
#     conda install h5py imageio jupyter matplotlib numpy tqdm
#     ```
# 
# 5. Install either the CPU version or GPU version of TensorFlow:
# 
#     CPU: *(Recommended as sufficient for this class)*
#     ```Shell
#     pip install --ignore-installed --upgrade tensorflow
#     ```
#     
#     GPU: (You'll also need to install some [Nvidia software](https://www.tensorflow.org/install/install_linux#nvidia_requirements_to_run_tensorflow_with_gpu_support))
#     ```Shell
#     pip install --ignore-installed --upgrade tensorflow-gpu
#     ```
# 
#     *Note: If you have GPUs available, you should use the GPU version for any serious research, as it is often 10-25x faster. For the purposes of this demo though, the CPU version is sufficent.* 

# ## Windows

# 1. Download and install Anaconda
# 
#     1.1 Go [here](https://www.continuum.io/downloads#windows) and click on the Python 3.6 installer matching your system's architecture (likely 64-bit X86).
#     
#     1.2 Open the installer from your downloads folder and run it by double clicking on it. 
#     
#     1.3 Follow the installer's instructions. Make sure to enable the option to make Anaconda your default Python.
#     
# 2. Open a Command Prompt and create a conda environment for TensorFlow:
# 
#     ```Shell
#     conda create -n tensorflow python=3.6
#     ```
#    
# 3. Activate your new environment:
# 
#     ```Shell
#     activate tensorflow
#     ```
#     
#     You will see your prompt prepend `(tensorflow)` to indicate that you are in your `tensorflow` environment.
# 
# 4. Install some supporting dependencies
#     ```Shell
#     conda install h5py imageio jupyter matplotlib numpy tqdm
#     ```
# 
# 5. Install either the CPU version or GPU version of TensorFlow:
# 
#     CPU: *(Recommended as sufficient for this class)*
#     ```Shell
#     pip install --ignore-installed --upgrade tensorflow
#     ```
#     
#     GPU: (You'll also need to install some [Nvidia software](https://www.tensorflow.org/install/install_windows#requirements_to_run_tensorflow_with_gpu_support)
#     ```Shell
#     pip install --ignore-installed --upgrade tensorflow-gpu
#     ```
# 
#     *Note: If you have GPUs available, you should use the GPU version for any serious research, as it is often 10-25x faster. For the purposes of this demo though, the CPU version is sufficent.*

# ## Mac OS X

# 1. Download and install Anaconda
# 
#     1.1 Go [here](https://www.continuum.io/downloads#macos) and click on the command line Python 3.6 installer.
#     
#     1.2 Navigate to your downloads directory in a terminal and run the installer:
#     
#     ```Shell
#     cd ~/Downloads/
#     bash Anaconda3-5.2.0-MacOSX-x86_64.sh
#     ```
# 
# 2. In a terminal, create a conda environment for TensorFlow:
# 
#     ```Shell
#     conda create -n tensorflow python=3.6
#     ```
#     
# 3. Activate your new environment:
# 
#     ```Shell
#     source activate tensorflow
#     ```
#     
#     You will see your prompt prepend `(tensorflow)` to indicate that you are in your `tensorflow` environment.
# 
# 4. Install some supporting dependencies
#     ```Shell
#     conda install h5py imageio jupyter matplotlib numpy tqdm
#     ```
# 
# 5. Install either the CPU version or GPU version of TensorFlow:
# 
#     CPU:
#     ```Shell
#     pip install --ignore-installed --upgrade tensorflow
#     ```
#     
#     GPU: Unfortunately, TensorFlow with GPU compatibility is no longer supported on Mac OS X
# 
#     *Note: If you have them available, you should use a machine with GPUs, as it is often 10-25x faster. For the purposes of this demo though, the CPU version is sufficent.*

# ## Verify TensorFlow Installation

# 1. While in your `tensorflow` environment, enter the Python shell with:
#     
#     ```Shell
#     python
#     ```
#     
#     Verify that your python version says Python 3.6.[x].
#     
# 2. Enter this program into your Python shell:
# 
#     ```Shell
#     import tensorflow as tf
#     hello = tf.constant('Hello, TensorFlow!')
#     sess = tf.Session()
#     print(sess.run(hello))
#     ```
#     
#     Possible warnings that `The TensorFlow library wasn't compiled to use * instructions` can be ignored. These warnings state that building from source can lead to speed improvements, but these won't make a dramatic difference for these demos.
#     
#     Your command line should return:
#     
#     ```Shell
#     Hello, TensorFlow!
#     ```   

# In[ ]:




