# Capstone-Project: Predict Future Stock Prices
## Machine Learning project to predict future stock prices using historical data and technical indicators.

 ### 1. The following python libraries and dependencies are required:
  keras  
  theano*  
  numpy  
  scipy  
  pandas  
  matplotlib  
  sklearn  
  statsmodels  
  IPython  
  G++ compiler  

 ### 2. *The installation of keras and theano will run as is, but it will be EXTREMELY slow without the G++ compiler installation.
  Installing the G++ compiler is hard.
  The documentation is here, but its not great:
  http://deeplearning.net/software/theano/install.html install

 ### 3. There are 2 python scripts which are used to implement this model:
  a. KerasPredict_Master_MultiOutput.py
  b. FeatureGeneration.py

  The main program/script which produces the plots & results is:
  KerasPredict_Master_MultiOutput.py

 ### 4. To run the KerasPredict_Master_MultiOutput.py script and see the outputs (charts, print outs),
  simply go to the very bottom of the file where this block is listed:

  if __name__=='__main__':
       fnRunSPY2015(False,9,1)
       fnRunOIL(False,9,1)
       fnRunVXX(False,9,5)
      fnRunQQQ(False,9,1)

  Uncomment out the function corresponding to the ETF you want to run:

  To run the prediction for the ETF: SPY, uncomment and run fnRunSPY2015(False,5,1)
  To run the prediction for the ETF: OIL, uncomment and run fnRunOIL(False,5,1)
  To run the prediction for the ETF: QQQ, uncomment and run fnRunQQQ(False,5,1)
  To run the prediction for the ETF: VXX, uncomment and run fnRunVXX(False,5,1)

  The FIRST parameter dictates whether a grid search should be run FIRST, before running the model.

  This grid search is very time consuming (e.g., hours)! It's advised to set this parameter to false untilyou are ready to truly ready   to run the grid search.

  The grid search results are reported under the heading "-----Grid Search Result Score and Parameters-----".

  The 2nd parm is the lookback window in days, and the 3rd param is the forecast horizon in days.

  To change the batch size or learning rate,  change the "learn_rate" and pBatchSize parameters within each of the following functions:
  fnRunSPY2015
  fnRunOIL
  fnRunVXX
  fnRunQQQ

 For example: 
   def fnRunQQQ(pBlnGridSearch =False,pLook_Back=10, pHorizon=1):
       '''Run the RNN predictions on the QQQ ETF using dates below
       Parm: pBlnGridSearch -true if we perform grid searching, false if not
       pLook_Back -number of days to look back within the model
       pHorizon -days ahead to forecast
       '''
        global lstrStock
       global learn_rate
       lStartDateTrain=datetime.date(1999, 3, 10)
       lEndDateTrain=datetime.date(2000, 1  , 31)
   
       lStartDateTest=datetime.date(2000, 2, 1)
       lEndDateTest=datetime.date(2000, 7  , 31)
       lstrStock="QQQ"
       learn_rate=0.0008   TO DO CHANGE THIS VALUE   
    
       fnMain(lstrStock,lStartDateTrain,lEndDateTrain, lStartDateTest,  lEndDateTest,None,pBlnGridSearch,\
              pLearnRate=learn_rate, pDropout=.1, pLayers=1, pNeuronMultiplier=1,\
              pLook_Back=pLook_Back, pHorizon=pHorizon,pBatchSize =8\  TO DO CHANGE THIS VALUE
    8, pEpochs=500)
