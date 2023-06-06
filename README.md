# CSE455 Final Project: Bird Classification Competition

### Problem Description 
We are trying to train our model to classify images of birds by their species.  

 ### Previous Work  
We used a pretrained model—[ResNeXt101 WSL](https://pytorch.org/hub/facebookresearch_WSL-Images_resnext/). This model is by Facebook AI, and it has been trained and fine-tuned on 940 million public images.   

### Our Approach  
To start off, we load the birds train dataset. We then apply three transforms, resizing, converting to tensors, and normalizing, to the images. Next, we split the dataset 80-20 into training and validation datasets. Then, we load in our pretrained model and freeze the model because we don't want to wipe the pretrained weights (because then there's no point in using a pretrained model). In terms of picking the model, we initially started with ResNet50 because we saw online that it works well for image classification. After getting lower validation accuracies than expected, we looked into other models. We found that ResNeXt101 WSL is supposed to work better than ResNet50. One downside is that it is much slower; however, we think that the tradeoff is worth it. Importantly, we need to modify the last fully connected layer to match the number of bird classes (otherwise we get a major error). Before evaluating, we need to define the loss function, so we can gauge the error between the predicted output and the target value. Finally, we loop through the training and validation phases a set number of times (this is the number of epoches), and we can output the losses and accuracies from each epoch. It's important to note, we are also trying to run all this on CUDA to speed up the computing on the GPU. 

### Datasets  
We used the birds dataset under the Kaggle competition. This dataset contains images of 555 different bird species. 

### Results
Here are our results:    
Epoch \[1/10], Val Loss: 2.5016, Val Accuracy: 0.4240  
Epoch \[2/10], Val Loss: 2.2090, Val Accuracy: 0.4815  
Epoch \[3/10], Val Loss: 2.1401, Val Accuracy: 0.5106  
Epoch \[4/10], Val Loss: 2.1616, Val Accuracy: 0.5287  
Epoch \[5/10], Val Loss: 2.2880, Val Accuracy: 0.5312  
Epoch \[6/10], Val Loss: 2.3541, Val Accuracy: 0.5297  
Epoch \[7/10], Val Loss: 2.3648, Val Accuracy: 0.5433  
Epoch \[8/10], Val Loss: 2.5361, Val Accuracy: 0.5333  
Epoch \[9/10], Val Loss: 2.4037, Val Accuracy: 0.5523  
Epoch \[10/10], Val Loss: 2.5200, Val Accuracy: 0.5403  

### Discussion

* **What problems did you encounter?**  
We encountered quite a few bugs when writing the code for this project. One problem that took us a long time to debug was that we passed the wrong  value for out_features into nn.Linear(). We were initially getting vague error messages, but later found online that setting os.environ\["CUDA_LAUNCH_BLOCKING"] = "1" gave more detailed stack traces. Through this, we were able to step through our code line by line and see where the error was coming from. Generally, our code took quite a while to run, so sometimes, we would wait upwards of 10-15 minutes, only to see that the outputted accuracy was shockingly low because we tweaked a line of code incorrectly. 

* **Are there next steps you would take if you kept working on the project?**  
If we kept working on the project, we would definitely want to increase the accuracy of our model. Additionally, if we had a lot of extra time, it might even be fun to try training a model from scratch!

* **How does your approach differ from others? Was that beneficial?**  
We're not super sure if our apporach differs from others. However, it does seem like ResNeXt101 WSL is less common than ResNet50. In our code, using ResNeXt101 WSL was beneficial, as we were able to achieve a higher accuracy with that over using ResNet50.

### Video

### Kaggle Notebook
