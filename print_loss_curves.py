import pandas as pd
import matplotlib.pyplot as plt

LOG_PATH = "./log_path"
SUB_FOLDER_PATH = LOG_PATH + "/training_100_epoch_torch_LSTM"
TRAINING_LOG_PATH = SUB_FOLDER_PATH + "/training.csv"
TESTING_LOG_PATH = SUB_FOLDER_PATH + "/testing.csv"
VALIDATION_LOG_PATH = SUB_FOLDER_PATH + "/validation.csv"
training_log_headers = [" epoch ", " batch " ," loss " , " ppl "]
testing_log_headers = [" select model path " ," loss " , " ppl "]
validation_log_headers = [" samples after epoch " ," loss " , " ppl "]
training_df = pd.read_csv(TRAINING_LOG_PATH,names = training_log_headers,skiprows=1)
testing_df = pd.read_csv(TESTING_LOG_PATH,names = testing_log_headers,skiprows=1)
validation_df = pd.read_csv(VALIDATION_LOG_PATH,names = validation_log_headers,skiprows=1)
training_epoch = list(training_df[" epoch "])
training_loss = list(training_df[" loss "])
training_ppl = list(training_df[" ppl "])
validation_epoch = list(validation_df[" samples after epoch "])
validation_loss = list(validation_df[" loss "])
validation_ppl = list(validation_df[" ppl "])
# testing_epoch = list(testing_df[" samples after epoch "])
testing_loss = list(testing_df[" loss "])
testing_ppl = list(testing_df[" ppl "])
# plt.title("My LSTM training ppl")
plt.title("Torch LSTM training ppl")
plt.plot(training_epoch,training_ppl)
plt.axis([0,100,0,900])
# plt.axis("off")
plt.savefig("./Torch_LSTM_training_ppl.png")
plt.show()


# plt.title("My LSTM validation ppl")
plt.title("Torch LSTM validation ppl")
plt.plot(validation_epoch,validation_ppl)
# plt.axis([0,100,300,180000])
plt.axis([0,100,300,1000])
# plt.axis([0,100,300,220000])
# plt.axis("off")
plt.savefig("./Torch_LSTM_validation_ppl.png")
plt.show()
