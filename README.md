# NLP LSTM
***Zoe's NLP homework , writing an LSTM structure***
* conda environment
```
ubuntu 18.04
cuda V10.0.130
cudnn v7.4.2.24
```
*upload the "NLP.tar.gz" file to your machine and run command below:*
```
mkdir ~/anaconda3/envs/NLP 
tar -zxvf NLP.tar.gz -C ~/anaconda3/envs/NLP
conda activate NLP
```

* Run</br>
using code below to train
```
python train.py
```
*and then you can get models in</br>*
```
/models/my_LSTM or /models/torch_LSTM
```
*get output logs in*
```
/log_path/training_100_epoch_My_LSTM or /log_path/training_100_epoch_torch_LSTM
```
*you can set whatever you like*

* Results</br>
*you can print curves by using*
```
python print_loss_curves.py
```
*you need to set the log path "SUB_FOLDER_PATH" to your log folder name*</br>
*and as for me , I get MyLSTM curves and TorchLSTM curves in validation set and training set , which you can see the results below*
 <div align=center>
 <img src="https://github.com/ZOUYAYI/NLP_LSTM/blob/main/Torch_LSTM_training_ppl.png" />
 <img src="https://github.com/ZOUYAYI/NLP_LSTM/blob/main/MY_LSTM_training_ppl.png" />
 <img src="https://github.com/ZOUYAYI/NLP_LSTM/blob/main/Torch_LSTM_validation_ppl.png" />
 <img src="https://github.com/ZOUYAYI/NLP_LSTM/blob/main/MY_LSTM_validation_ppl.png" />
 </div>
