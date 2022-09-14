# Here we have commonly used functions. Hopefully this will be useful for others hoping to use our results

from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from joblib import dump, load
from scipy import stats
from sklearn.preprocessing import StandardScaler
import os
import random



def train_EN(CpG_df, ages_df, data_name, test_size=0.2, random_state=42, l1_ratio = 0.5, n_alphas = 50, cv = 10, n_jobs=11,
             max_iter=5000, tol = 0.001, selection='cyclic'):

  #In this function we take the CpG dataframe, age dataframe, and data name as a string and return a trained
  #Elastic net model. The charectristics of the test train split and Elastic Net model can also be passed.
  #Where this model is saved and the prfix affecting it's name are currently hardset


  methyl_raw_train, methyl_raw_test, age_train, age_test = train_test_split(CpG_df, ages_df, test_size=test_size, random_state=random_state)

  #Read in age training data as a list
  Y_train=age_train.values.ravel()

  #Scale our data such that the fit is to the training set
  scaler = StandardScaler().fit(methyl_raw_train)
  methyl_train = scaler.transform(methyl_raw_train)

  methyl_test = scaler.transform(methyl_raw_test)


  #train elasticnet
  #Create Elastic Net model
  elastic_netCV = ElasticNetCV(l1_ratio = 0.5, n_alphas = 50, cv = 10, n_jobs=11, random_state = random_state,
                            max_iter=5000, tol = 0.001, selection='cyclic')

  elastic_netCV.fit(methyl_train,Y_train)

  if os.path.exists("Trained_Models") == False:
    os.mkdir("Trained_Models")

  #Save the Elastic Net model
  dump(elastic_netCV, "Trained_Models\\elastic_netCV_"+str(data_name)+".joblib")

  return elastic_netCV



def Find_R_Clocks(CpG_df, ages_df, data_name, coeffs_df):
     #In this function we take the CpG dataframe, age dataframe, a dataframe with Clock CpG coefficients and
     # data name as a string and return a dataframe  with the wheight of each clock CPG in the model, the rank of CpG by magnitude,
     # and Pearson's R^2 value. Components of the name and directory in which the R^2 values are
     #saved are hardset by the model. Warning! TO save R's coherently then coeffs_df must be sorted by rank with the
     #first row being rank 1.

    r_values =[]

    #generate a list of all CpG names
    Clock_CpGs = coeffs_df["CpG"]

    Clock_CpGs

    coeff_R_df = coeffs_df.copy()
    for item in coeffs_df.index:
      cpg = CpG_df.T.iloc[item].astype(float)
      regression = stats.linregress(cpg, ages_df.Age.astype(float))
      slope, intercept, rvalue, pvalue, stderr = regression
      r_values.append(rvalue**2)


    coeff_R_df["Rvalue"] = r_values
    Clock_CpGs["Rvalue"] = r_values

    if os.path.exists("CpG_Rs") == False:
      os.mkdir("CpG_Rs")

    #save the R^2 value of each Clock CpG
    Clock_CpGs.to_pickle('CpG_Rs\CpG_R_' +str(data_name)+'_clocks.pkl')

    return coeff_R_df

def Find_R(CpG_df, ages_df, data_name):
     #In this function we take the CpG dataframe, age dataframe, and
     # data name as a string and return a dataframe  with the CpG name
     # and Pearson's R^2 value of all CpGs in the dataset. Components of the name and directory in
     # which the R^2 values are saved are hardset by the model.
     #BE Warned: This takes a very long time to run

    #generate a list of all CpG names
    All_CpGs = pd.DataFrame(CpG_df.columns)
    All_CpGs.rename(columns={All_CpGs.columns[0]: 'CpG'})
    All_CpGs

    r_values =[]
    i = 0

    CpG_Stats = All_CpGs.copy()
    for item in All_CpGs.index:
        cpg = CpG_df.T.iloc[item].astype(float)
        regression = stats.linregress(cpg, ages_df.Age.astype(float))
        slope, intercept, rvalue, pvalue, stderr = regression
        r_values.append(rvalue**2)
        i = i+1
        if i % 100 == 0:
            print(i)

    CpG_Stats["Rvalue"] = r_values

    if os.path.exists("CpG_Rs") == False:
      os.mkdir("CpG_Rs")

    #save the R^2 value of each Clock CpG
    CpG_Stats.to_pickle('CpG_Rs\CpG_R_' +str(data_name)+'.pkl')

    return CpG_Stats

def remove_nonsig(CpG_df, ages_df, data_name, trained_model, maximum_removed=100, step=10, starting_state = 0 , test_size=0.2, random_state=42, l1_ratio = 0.5,
                  n_alphas = 50, cv = 10, n_jobs=11, max_iter=5000, tol = 0.001, selection='cyclic'):
    #function which takes the CpG dataframe, age dataframe,
    # data name as a string, and the previously tained model on the whole dataset, the maximum
    # percent of non original clock CpGs we wish to remove and the step size, and where we would like to begin removing from
    #(defualt is zero so the first retrained model will have a % of non clock CpGs removed equal to step size)  The function
    #saves each resulting model. Wanring this is very heavy on RAm use and python can be bad about clearing ram.
    #So you may have to move in smaller ranges then you would like

    coeffs_original = pd.DataFrame(trained_model.coef_)
    coeffs_original = coeffs_original[(coeffs_original.T != 0).any()]
    coeffs_original = coeffs_original.rename(columns={coeffs_original.columns[0]: 'Magnitude'})

    colnames = pd.DataFrame(CpG_df.columns)
    sig_cpgs_original = colnames.iloc[coeffs_original.index]

    # Create a list of all non clock CpGs
    first = True
    for cpg in sig_cpgs_original["CpG"].to_list():

        if first == True:
            nonsig_cpgs_original = colnames[colnames["CpG"].str.contains(cpg)==False]
            first = False
        else:
            nonsig_cpgs_original = nonsig_cpgs_original[nonsig_cpgs_original["CpG"].str.contains(cpg)==False]

    #Find length of non clock CpGs
    len_nonsig_original = len(nonsig_cpgs_original)

    for i in range((maximum_removed-starting_state)/step):

        #Find number to be removed
        num_removed = int(len_nonsig_original*0.01*(((1+i)*step)+starting_state))

        #Generate CpGs to be removed
        list_removed = random.sample(range(0, len_nonsig_original), num_removed)
        cpgs_removed = nonsig_cpgs_original.iloc[list_removed]

        #Generate a new data set with the randomly selected CpGs removed
        CpG_df_removed = CpG_df.drop(cpgs_removed["CpG"].to_list(), axis=1)

        train_EN(CpG_df_removed, ages_df, str(data_name) +"_"+str((((1+i)*step)+starting_state))+"%_nonclock_CpGs_removed")
