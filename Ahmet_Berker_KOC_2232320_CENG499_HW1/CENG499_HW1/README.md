1. To start the training, write 'python train.py' command in terminal
    This code train all models with different hyper-parameters thanks to loops and create 114 different model
    model folder have these models but I only put the best model in my zip file due to size. Ir creates acc_loss.txt file and write the avarage loss and accuracy of each models. 
    Model number and related parameter is explained in report.
2. To start the training, write 'python test.py' command in terminal
    This code test the best model by using test set. It uses the dataset_set to obtain data. It create the test_label.txt
    test_result.png is the result from the http://207.154.196.239/
3. To start the training, write 'python plot.py' command in terminal. 
    This code train again and plots the training and validation losses of the best model. Figure_1 is the plot.
4. Data folder must be in the same file with these code files. There is no data in the zip due to size.