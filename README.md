# ECE-542-Competition
Competition project for ECE 542


### Setup
* If online Clone this repository  
`git clone https://github.ncsu.edu/george2/ECE-542-Competition.git`  
* Dependecies: `python3` and `pip`
* Install requirements  
`pip install -r requirements.txt`  
* Copy `properties_dummy.py` to `properties.py` and update `DATA_HOME` and `TEST_HOME`  


### Execution
##### Training
* Window size can be in [1, 2, 4, 6, 10, 30, 60]  
`python runner.py --mode train  --window <WindowSize>`  
* Model will be saved in `models/successful_lstm_<WindowSize>.mdl`


##### Prediction
* Window size can be in [1, 2, 4, 6, 10, 30, 60]  
`python runner.py --mode predict  --window <WindowSize>`  
* Results will be printed in the command prompt and confusion matrix will be saved in `results/successful_lstm_<WindowSize>/`
  
  
  
### CodeBase
* The experimented models are described in `model_utils.py`
* The data streamer is described in `data_utils.py`
* The experiments are run in `runner.py`
* We also have an IPython notebook which runs the same in `ECE542_Competition_Final.ipynb`
