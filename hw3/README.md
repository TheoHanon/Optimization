# LINMA 2471 : HW3
Here you can find the implementation of a short & long step path following method. We leverage those methods to train a linear classifier. We provide plots and relevant statistics of the classifier after the training.
## Structure of the directory
The implementation is spread in different files:
* `derivative.py` contains the computation of the gradient and the hessian of the objective function.
* `tools.py` provides different type of function: some are usefull for the implementation of the long & short step, the other functions are tools to import the data and compute the accuracy. 
* `hW3_opti.ipynb` follows the thread of the homework. We use PCA to speed up the process so that you can check that each question are properly implemented in a short period of time (depending of your computer running the whole notebook should take ~10 minutes). Note that the figures of the report have been computed via this notebook but the plots you will see are different since we tuned the parameter to run the notebook in few minutes instead of hours.
## Getting started
### Requirement
You need to following packages:
* `sklearn`
* `numpy`
* `matplotlib`
* `alive_progress` - A progress bar package with cool animations and real-time stats. More information and installation instructions can be found at [alive-progress PyPI page](https://pypi.org/project/alive-progress/).
### Run the code 
You can run the code in the notebook. As previously mention it follows the structure of the homework. We created separate cells for the implementation of and the results each question. 

### Warning
Depending of the set of parameter two types of Warning can appear : `LinAlgWarning` and `RuntimeWarning`, we didn't shut them off in the console. However they do not impact the working of the code it just impacts the accuracy of the final results. They are caused by rounding errors and bad conditionning. Finally note that we made our code robust to errors thus the algorithm always finishes but again the result might not be accurate. 
