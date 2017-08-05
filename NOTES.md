## NOTES


- How to determine maximum batch size?
- What are the indications that training is going well or badly?
- Is there any down side to my approach with validation script?
- How would i incorporate some hdf5 saving into my workflow?

#OVERFITTING
- You will notice if training accuracy is much higher than validation


#T1
- After 13 epochs accuracy stops improving
- Accuracy is very low (0.362)
- After 2nd epoch validation accuracy freezes at 0.3360
- during training the accuracy seemed much higher (~80) but the summaries tell a different story

#T2 LOGS :sam-cam
1920/1920 [==============================] - 33s - loss: 0.6843 - acc: 0.5760 - val_loss: 2.6257 - val_acc: 0.5125                                                            
Epoch 2/50                                                                                                                                                                    
1920/1920 [==============================] - 30s - loss: 0.6630 - acc: 0.6250 - val_loss: 3.9856 - val_acc: 0.5083                                                            
Epoch 3/50                                                                                                                                                                    
1920/1920 [==============================] - 30s - loss: 0.6450 - acc: 0.6552 - val_loss: 5.0728 - val_acc: 0.5021                                                            
Epoch 4/50                                                                                                                                                                    
1920/1920 [==============================] - 30s - loss: 0.6335 - acc: 0.6552 - val_loss: 5.7212 - val_acc: 0.5021                                                            
Epoch 5/50                                                                                                                                                                    
1920/1920 [==============================] - 30s - loss: 0.6233 - acc: 0.6745 - val_loss: 6.1444 - val_acc: 0.5021                                                            
Epoch 6/50                                                                                                                                                                    
1920/1920 [==============================] - 30s - loss: 0.6132 - acc: 0.6875 - val_loss: 6.4218 - val_acc: 0.5021                                                            
Epoch 7/50                                                                                                                                                                    
1920/1920 [==============================] - 30s - loss: 0.6050 - acc: 0.6906 - val_loss: 6.6029 - val_acc: 0.5021                                                            
Epoch 8/50                                                                                                                                                                    
1920/1920 [==============================] - 32s - loss: 0.5975 - acc: 0.6995 - val_loss: 6.7176 - val_acc: 0.5021                                                            
Epoch 9/50                                                                                                                                                                    
1920/1920 [==============================] - 31s - loss: 0.5917 - acc: 0.6979 - val_loss: 6.7884 - val_acc: 0.5021                                                            
Epoch 10/50                                
1920/1920 [==============================] - 30s - loss: 0.5845 - acc: 0.7177 - val_loss: 6.8826 - val_acc: 0.5042                                                            
Epoch 11/50                                
1920/1920 [==============================] - 30s - loss: 0.5778 - acc: 0.7135 - val_loss: 6.9231 - val_acc: 0.5062                                                            
Epoch 12/50                                
1920/1920 [==============================] - 30s - loss: 0.5733 - acc: 0.7198 - val_loss: 6.9043 - val_acc: 0.5083                                                            
Epoch 13/50                                
1920/1920 [==============================] - 30s - loss: 0.5681 - acc: 0.7203 - val_loss: 6.9571 - val_acc: 0.5083                                                            
Epoch 14/50                                
1920/1920 [==============================] - 30s - loss: 0.5654 - acc: 0.7276 - val_loss: 6.9330 - val_acc: 0.5083                                                            
Epoch 15/50                                
1920/1920 [==============================] - 30s - loss: 0.5600 - acc: 0.7286 - val_loss: 6.9109 - val_acc: 0.5104                                                            
Epoch 16/50                                
1920/1920 [==============================] - 30s - loss: 0.5567 - acc: 0.7385 - val_loss: 6.9255 - val_acc: 0.5146                                                            
Epoch 17/50                                
1920/1920 [==============================] - 30s - loss: 0.5494 - acc: 0.7391 - val_loss: 6.9168 - val_acc: 0.5146     
Epoch 18/50                                                                            
1920/1920 [==============================] - 30s - loss: 0.5496 - acc: 0.7385 - val_loss: 6.8663 - val_acc: 0.5146                                                            
Epoch 19/50                                
1920/1920 [==============================] - 31s - loss: 0.5452 - acc: 0.7438 - val_loss: 6.8451 - val_acc: 0.5146                                                            
Epoch 20/50                                
1920/1920 [==============================] - 31s - loss: 0.5406 - acc: 0.7422 - val_loss: 6.8062 - val_acc: 0.5188         


#GITHUB Q

Hey,

I am trying to implement a version of your code that will allow for more than 2 classes, as well as larger datasets by using the fit_generator function.  I was able to get your code to work fine

However it seems that something I am doing is breaking the model, as even with 2 classes now, the validation accuracy seems to freeze and it doesn't get any better.

Here is my observations of the problems.
- After 13 epochs accuracy stops improving
- Accuracy is very low (0.362)
- After 2nd epoch validation accuracy freezes at 0.3360
- during training the accuracy seemed much higher (~80) but the summaries tell a different story


Here is a sample of my training logs:


