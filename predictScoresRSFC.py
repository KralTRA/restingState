import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import datasets, linear_model, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC


import warnings
warnings.filterwarnings("ignore")

####################################################################################################################
# data location if in the server
datapath="/study/strengthen_nccam/analysis/nccam3/rsfc/graph_theory/results/"
####################################################################################################################

####################################################################################################################
# Read datafile(s)
# phenotype  & a couple of relevant graph metrics (NA Factor, R Ce Amygdala WMD & PC)
# datafile="WMDdata_T1_MNPNA.csv"
# datafile="WMDdata_T1_MNP.csv"
# datafile="PCdata_T1_MNPNA.csv"

# gordon data (complete)
datafile="T1/T1_graph.csv"
groupDataFile="/study/nccam3/data/masterfile/unblinded/unblindedgrps_attendance.csv"

groupData=pd.read_csv(groupDataFile)
myData=pd.read_csv(datapath+datafile)

# remove MNPAs
# indexNames = myData[ (myData['Subject'] > 2999) & (myData['Subject'] < 4000) ].index
# myData.drop(indexNames , inplace=True)

# select only MNPNAs
indexNames = myData[ (myData['Subject'] > 2999) ].index
myData.drop(indexNames , inplace=True)

# filter for specific subject list
# myData = myData[~myData['Subject'].isin([2000,2006,2074,2077,2126,2138,2010,2030,2035,2037,2044,2050,2055,2080,2081,2097,2140,4002,4017,4019])]

# myData1=pd.read_csv(datapath+datafile1)
# myData2=pd.read_csv(datapath+datafile2)

# compute difference scores
# T2vsT1data=myData2.set_index('Subject').subtract(myData1.set_index('Subject')).reset_index()
# myData=T2vsT1data
# myData = myData[~myData['Subject'].isin([2000,2006,2074,2077,2126,2138,2010,2030,2031,2035,2037,2044,2050,2055,2064,2079,2080,2081,2097,2140,2142,4002,4017,4019])]

# a different method to filter
# mnp = y[~y['GroupR'].isin(['LTM'])]

## if datafile is Adj matrix, make sure no nans
# myData.replace({' ': np.nan}, inplace=True)
# myData=myData.dropna(axis=1, how='any')
####################################################################################################################
# Get X and y
# graph theory data

### Relations at T1
# all Gordon at T1
# X=myData[['GordonROIs_0','GordonROIs_1','GordonROIs_2','GordonROIs_3','GordonROIs_4','GordonROIs_5','GordonROIs_6','GordonROIs_7','GordonROIs_8','GordonROIs_9','GordonROIs_10','GordonROIs_11','GordonROIs_12','GordonROIs_13','GordonROIs_14','GordonROIs_15','GordonROIs_16','GordonROIs_17','GordonROIs_18','GordonROIs_19','GordonROIs_20','GordonROIs_21','GordonROIs_22','GordonROIs_23','GordonROIs_24','GordonROIs_25','GordonROIs_26','GordonROIs_27','GordonROIs_28','GordonROIs_29','GordonROIs_30','GordonROIs_31','GordonROIs_32','GordonROIs_33','GordonROIs_34','GordonROIs_35','GordonROIs_36','GordonROIs_37','GordonROIs_38','GordonROIs_39','GordonROIs_40','GordonROIs_41','GordonROIs_42','GordonROIs_43','GordonROIs_44','GordonROIs_45','GordonROIs_46','GordonROIs_47','GordonROIs_48','GordonROIs_49','GordonROIs_50','GordonROIs_51','GordonROIs_52','GordonROIs_53','GordonROIs_54','GordonROIs_55','GordonROIs_56','GordonROIs_57','GordonROIs_58','GordonROIs_59','GordonROIs_60','GordonROIs_61','GordonROIs_62','GordonROIs_63','GordonROIs_64','GordonROIs_65','GordonROIs_66','GordonROIs_67','GordonROIs_68','GordonROIs_69','GordonROIs_70','GordonROIs_71','GordonROIs_72','GordonROIs_73','GordonROIs_74','GordonROIs_75','GordonROIs_76','GordonROIs_77','GordonROIs_78','GordonROIs_79','GordonROIs_80','GordonROIs_81','GordonROIs_82','GordonROIs_83','GordonROIs_84','GordonROIs_85','GordonROIs_86','GordonROIs_87','GordonROIs_88','GordonROIs_89','GordonROIs_90','GordonROIs_91','GordonROIs_92','GordonROIs_93','GordonROIs_94','GordonROIs_95','GordonROIs_96','GordonROIs_97','GordonROIs_98','GordonROIs_99','GordonROIs_100','GordonROIs_101','GordonROIs_102','GordonROIs_103','GordonROIs_104','GordonROIs_105','GordonROIs_106','GordonROIs_107','GordonROIs_108','GordonROIs_109','GordonROIs_110','GordonROIs_111','GordonROIs_112','GordonROIs_113','GordonROIs_114','GordonROIs_115','GordonROIs_116','GordonROIs_117','GordonROIs_118','GordonROIs_119','GordonROIs_120','GordonROIs_121','GordonROIs_122','GordonROIs_123','GordonROIs_124','GordonROIs_125','GordonROIs_126','GordonROIs_127','GordonROIs_128','GordonROIs_129','GordonROIs_130','GordonROIs_131','GordonROIs_132','GordonROIs_133','GordonROIs_134','GordonROIs_135','GordonROIs_136','GordonROIs_137','GordonROIs_138','GordonROIs_139','GordonROIs_140','GordonROIs_141','GordonROIs_142','GordonROIs_143','GordonROIs_144','GordonROIs_145','GordonROIs_146','GordonROIs_147','GordonROIs_148','GordonROIs_149','GordonROIs_150','GordonROIs_151','GordonROIs_152','GordonROIs_153','GordonROIs_154','GordonROIs_155','GordonROIs_156','GordonROIs_157','GordonROIs_158','GordonROIs_159','GordonROIs_160','GordonROIs_161','GordonROIs_162','GordonROIs_163','GordonROIs_164','GordonROIs_165','GordonROIs_166','GordonROIs_167','GordonROIs_168','GordonROIs_169','GordonROIs_170','GordonROIs_171','GordonROIs_172','GordonROIs_173','GordonROIs_174','GordonROIs_175','GordonROIs_176','GordonROIs_177','GordonROIs_178','GordonROIs_179','GordonROIs_180','GordonROIs_181','GordonROIs_182','GordonROIs_183','GordonROIs_184','GordonROIs_185','GordonROIs_186','GordonROIs_187','GordonROIs_188','GordonROIs_189','GordonROIs_190','GordonROIs_191','GordonROIs_192','GordonROIs_193','GordonROIs_194','GordonROIs_195','GordonROIs_196','GordonROIs_197','GordonROIs_198','GordonROIs_199','GordonROIs_200','GordonROIs_201','GordonROIs_202','GordonROIs_203','GordonROIs_204','GordonROIs_205','GordonROIs_206','GordonROIs_207','GordonROIs_208','GordonROIs_209','GordonROIs_210','GordonROIs_211','GordonROIs_212','GordonROIs_213','GordonROIs_214','GordonROIs_215','GordonROIs_216','GordonROIs_217','GordonROIs_218','GordonROIs_219','GordonROIs_220','GordonROIs_221','GordonROIs_222','GordonROIs_223','GordonROIs_224','GordonROIs_225','GordonROIs_226','GordonROIs_227','GordonROIs_228','GordonROIs_229','GordonROIs_230','GordonROIs_231','GordonROIs_232','GordonROIs_233','GordonROIs_234','GordonROIs_235','GordonROIs_236','GordonROIs_237','GordonROIs_238','GordonROIs_239','GordonROIs_240','GordonROIs_241','GordonROIs_242','GordonROIs_243','GordonROIs_244','GordonROIs_245','GordonROIs_246','GordonROIs_247','GordonROIs_248','GordonROIs_249','GordonROIs_250','GordonROIs_251','GordonROIs_252','GordonROIs_253','GordonROIs_254','GordonROIs_255','GordonROIs_256','GordonROIs_257','GordonROIs_258','GordonROIs_259','GordonROIs_260','GordonROIs_261','GordonROIs_262','GordonROIs_263','GordonROIs_264','GordonROIs_265','GordonROIs_266','GordonROIs_267','GordonROIs_268','GordonROIs_269','GordonROIs_270','GordonROIs_271','GordonROIs_272','GordonROIs_273','GordonROIs_274','GordonROIs_275','GordonROIs_276','GordonROIs_277','GordonROIs_278','GordonROIs_279','GordonROIs_280','GordonROIs_281','GordonROIs_282','GordonROIs_283','GordonROIs_284','GordonROIs_285','GordonROIs_286','GordonROIs_287','GordonROIs_288','GordonROIs_289','GordonROIs_290','GordonROIs_291','GordonROIs_292','GordonROIs_293','GordonROIs_294','GordonROIs_295','GordonROIs_296','GordonROIs_297','GordonROIs_298','GordonROIs_299','GordonROIs_300','GordonROIs_301','GordonROIs_302','GordonROIs_303','GordonROIs_304','GordonROIs_305','GordonROIs_306','GordonROIs_307','GordonROIs_308','GordonROIs_309','GordonROIs_310','GordonROIs_311','GordonROIs_312','GordonROIs_313','GordonROIs_314','GordonROIs_315','GordonROIs_316','GordonROIs_317','GordonROIs_318','GordonROIs_319','GordonROIs_320','GordonROIs_321','GordonROIs_322','GordonROIs_323','GordonROIs_324','GordonROIs_325','GordonROIs_326','GordonROIs_327','GordonROIs_328','GordonROIs_329','GordonROIs_330','GordonROIs_331','GordonROIs_332','GordonROIs_333','GordonROIs_334','GordonROIs_335','GordonROIs_336','GordonROIs_337','GordonROIs_338','GordonROIs_339','GordonROIs_340','GordonROIs_341','GordonROIs_342','GordonROIs_343','GordonROIs_344','GordonROIs_345','GordonROIs_346','GordonROIs_347','GordonROIs_348','GordonROIs_349','GordonROIs_350']]
# 
# # Gordon Default at T1 --- # 80% accy for detecting LTM vs MNP PC at T1 (chance = 76.8%)
# X=myData[['GordonROIs_0','GordonROIs_3','GordonROIs_5','GordonROIs_24','GordonROIs_25','GordonROIs_43','GordonROIs_93','GordonROIs_113','GordonROIs_115','GordonROIs_116','GordonROIs_125','GordonROIs_126','GordonROIs_144','GordonROIs_145','GordonROIs_149','GordonROIs_150','GordonROIs_151','GordonROIs_153','GordonROIs_155','GordonROIs_156','GordonROIs_161','GordonROIs_164','GordonROIs_183','GordonROIs_185','GordonROIs_199','GordonROIs_219','GordonROIs_224','GordonROIs_256','GordonROIs_258','GordonROIs_277','GordonROIs_278','GordonROIs_289','GordonROIs_314','GordonROIs_315','GordonROIs_320','GordonROIs_321','GordonROIs_322','GordonROIs_323','GordonROIs_324','GordonROIs_325','GordonROIs_330']]
# 
# # Gordon DorsalAtt - up to 81% predictive for LTMvsMNP
# X=myData[['GordonROIs_40','GordonROIs_41','GordonROIs_42','GordonROIs_48','GordonROIs_50','GordonROIs_51','GordonROIs_54','GordonROIs_73','GordonROIs_86','GordonROIs_87','GordonROIs_90','GordonROIs_91','GordonROIs_94','GordonROIs_99','GordonROIs_105','GordonROIs_106','GordonROIs_109','GordonROIs_112','GordonROIs_154','GordonROIs_188','GordonROIs_198','GordonROIs_202','GordonROIs_207','GordonROIs_210','GordonROIs_235','GordonROIs_249','GordonROIs_251','GordonROIs_252','GordonROIs_261','GordonROIs_265','GordonROIs_270','GordonROIs_274']]
# 
# # Gordon FrontoParietal - up to 84% predictive for LTMvsMNP
# X=myData[['GordonROIs_6','GordonROIs_8','GordonROIs_23','GordonROIs_77','GordonROIs_95','GordonROIs_107','GordonROIs_108','GordonROIs_147','GordonROIs_148','GordonROIs_166','GordonROIs_167','GordonROIs_169','GordonROIs_181','GordonROIs_239','GordonROIs_259','GordonROIs_260','GordonROIs_271','GordonROIs_272','GordonROIs_275','GordonROIs_276','GordonROIs_318','GordonROIs_319','GordonROIs_326','GordonROIs_327']]

# Gordon Salience at T1
# subset by network
myData=myData.merge(groupData, left_on=['Subject'], right_on=['id'], how='left')
myData=myData[myData['ROI'].str.contains("Salience")]
myData['GroupR']='MNP'
# myData.loc[myData['Group'] =='MM', 'GroupR'] = 'LTM'
# myData.loc[myData['Group'] =='LKCM', 'GroupR'] = 'LTM'

myData.dropna(subset=['PC'])
X=myData.pivot_table(index=['Subject'], values='PC', columns='ROI_num')
# y=myData.pivot_table(index=['Subject'], values='GroupR')

y=myData[['sr_t1_pwb_pwbTotal']]
# y=myData[['GroupR']]


####################################################################################################################


####################################################################################################################
# Creating linear regression object
regr = linear_model.LinearRegression()
# or other model types
regr = linear_model.Ridge (alpha = .5)
regr = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])
####################################################################################################################


####################################################################################################################
# Creating method of cross validation
# k fold (n=10) recommended. If dv binary, stratified sampling recommended? (equal proportion for binary classes); default in sklearn
kf = KFold(n_splits=10, shuffle=True, random_state=1) # kf.split(X) is the way to call it; yields train_index, test_index
# LOO
# loo = LeaveOneOut() # loo.split(X) is the way to call it; yields train_index, test_index
# Fixed Split ok, but prediction will have high variance: X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.3)
####################################################################################################################


####################################################################################################################
# Predicting w/ cross validation
y_pred= cross_val_predict(regr, X, y,  cv=kf.split(X) ) # or cv=10 

# Correlating predicted w/ actual y values
np.corrcoef(np.array(y),y_pred)

# Plotting predicted & actual y values
fig, ax = plt.subplots()
ax.scatter(y, y_pred)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
####################################################################################################################



####################################################   THE END   ###################################################

####################################################################################################################
# Creating SVM object / choose estimator
clf = svm.SVC(kernel='linear')

# split the data (80/20)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

# choose cross-validation, default = 10 splits
cv = ShuffleSplit(X_train.shape[0], test_size=0.2, random_state=0)

# apply cross-validation - tune parameters
gammas = np.logspace(-6, -1, 10)
classifier = GridSearchCV(estimator=clf, cv=cv, param_grid=dict(gamma=gammas))
classifier.fit(X_train, y_train)

# de-bug algorithm with learning curve
# title = 'Learning Curves (SVM, linear kernel, $\gamma=%.6f$)' %classifier.best_estimator_.gamma
# estimator = SVC(kernel='linear', gamma=classifier.best_estimator_.gamma)
# plot_learning_curve(estimator, title, X_train, y_train) # NOT WORKING - CANNOT IMPORT function
# plt.show()

# evaluate test set
classifier.score(X_test, y_test)

# train the model
#clf.fit(X_train,y_train)

# predict y for the test set
#y_pred = clf.predict(X_test)

# evaluate the model
#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#dec = clf.decision_function([[1]]) #?
####################################################################################################################


####################################################################################################################

####################################################################################################################
# Evaluating models w/ cross validation
# if scoring=mse, take the negative and square for RMSE// smaller is better obvi.
# if improving parameters, use this or from sklearn.grid_search import GridSearchCV
print (np.sqrt(-(cross_val_score(regr,X, y, cv=kf.split(X) , scoring="neg_mean_squared_error"))).mean())
# R*2; this is more interpretable but unstable.
scores = cross_val_score(regr, X, y, scoring='r2', cv=kf.split(X),n_jobs=1) #cv could be simply 10 for default K, or loo.split(X)
print (scores.mean())
####################################################################################################################

####################################################################################################################
# extra: if doing it step-by-step instead of using "cross_val_predict"
kf = KFold(n_splits=10) 
y_pred=[]

for train_index, test_index in kf.split(X):
 	X_train, X_test = X.ix[train_index], X.ix[test_index]
    y_train, y_test = y.ix[train_index], y.ix[test_index]
	regr.fit(X_train,  y_train)
	y_pred.extend(regr.predict(X_test))
	
#correl predicted and actual y	
np.corrcoef(np.array(y_pred),y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)
print("Mean squared error: %.2f"% np.mean((regr.predict(X_test) - y_test) ** 2))
print('Variance score: %.2f' % regr.score(X_test, y_test)) # 1 is perfect
####################################################################################################################
