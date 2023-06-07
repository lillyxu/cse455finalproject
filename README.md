# CSE455 Final Project: Bird Classification Competition

### Problem Description 
We are trying to train our model to classify images of birds by their species.  

 ### Previous Work  
We used a pretrained model—[ResNeXt101 WSL](https://pytorch.org/hub/facebookresearch_WSL-Images_resnext/). This model is by Facebook AI, and it has been trained and fine-tuned on 940 million public images.   

### Our Approach  
To start off, we load the birds train dataset. We then apply three transforms, resizing, converting to tensors, and normalizing, to the images. Next, we split the dataset 90-10 into training and validation datasets. Then, we load in our pretrained model and freeze the model because we don't want to wipe the pretrained weights. In terms of picking the model, we initially started with ResNet50 because we saw online that it works well for image classification. After getting lower validation accuracies than expected, we looked into other models. We found that ResNeXt101 WSL is supposed to work better than ResNet50. One downside is that it is much slower; however, we think that the tradeoff is worth it. Importantly, we need to modify the last fully connected layer to match the number of bird classes (otherwise we get a major error), and optimize on that layer using Adam. This optimizer is supposed to perform better than SGD, and again, it is a little slower, but we wanted to prioritize accuracy. Before evaluating, we need to define the loss function, so we can gauge the error between the predicted output and the target value. Finally, we loop through the training and validation phases a set number of times (this is the number of epoches), and we can output the losses and accuracies from each epoch. At the end of each epoch, we store the model if it performed better than the previous best epoch's because we noticed that sometimes later epochs performed worse than earlier ones. We are then able to choose the best-performing model to predict against the test dataset. It's important to note, we are also trying to run all this on CUDA to speed up the computing on the GPU. 

### Datasets  
We used the birds dataset under the Kaggle competition. This dataset contains images of 555 different bird species. 

### Results
Here are our results:    
Epoch \[1/10], Val Loss: 2.4402, Val Accuracy: 0.4367
Epoch \[2/10], Val Loss: 2.3585, Val Accuracy: 0.4754
Epoch \[3/10], Val Loss: 2.2171, Val Accuracy: 0.5161
Epoch \[4/10], Val Loss: 2.3098, Val Accuracy: 0.5288
Epoch \[5/10], Val Loss: 2.2605, Val Accuracy: 0.5379
Epoch \[6/10], Val Loss: 2.2387, Val Accuracy: 0.5573
Epoch \[7/10], Val Loss: 2.3215, Val Accuracy: 0.5545
Epoch \[8/10], Val Loss: 2.3422, Val Accuracy: 0.5617
Epoch \[9/10], Val Loss: 2.4347, Val Accuracy: 0.5576
Epoch \[10/10], Val Loss: 2.5601, Val Accuracy: 0.5615  

### Discussion

* **What problems did you encounter?**  
We encountered quite a few bugs when writing the code for this project. One problem that took us a long time to debug was that we passed the wrong  value for out_features into nn.Linear(). We were initially getting vague error messages, but later found online that setting os.environ\["CUDA_LAUNCH_BLOCKING"] = "1" gave more detailed stack traces. Through this, we were able to step through our code line by line and see where the error was coming from. Another bug we encountered was that we were freezing the model's parameters and then creating an optimizer on those parameters. However, since those parameters were frozen, it didn't make sense to optimize on them. Instead, we should optimize on the fully connected layer we added to our model. Generally, our code took quite a while to run, so sometimes, we would wait upwards of 10-15 minutes, only to see that the outputted accuracy was shockingly low because we tweaked a line of code incorrectly. 

* **Are there next steps you would take if you kept working on the project?**  
If we kept working on the project, we would definitely want to increase the accuracy of our model. To do this, we would want to add additional layers and more epochs to our model. We could also look into further reducing the size of the validation dataset, so that we have more data to train the model on. We would also want to decrease the loss of our model and avoid overfitting. We could do this by applying more random image transforms to make the model more generalized. Additionally, if we had a lot of extra time, it might even be fun to try training a model from scratch!

* **How does your approach differ from others? Was that beneficial?**  
We chose to split the training dataset into two so that we had a small percentage of it reserved for model validation. This helped us evaluate the model at each epoch and store the best-performing model state. Furthermore, we optimized by storing the best-performing model state because we often had the case where previous models performed better than later ones. It also seems like ResNeXt101 WSL is less common than ResNet50. In our code, using ResNeXt101 WSL was beneficial, as we were able to achieve a higher accuracy with that over using ResNet50.

### Video  
https://youtu.be/w9arXYzSbQc

### Kaggle Notebook
