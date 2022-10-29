# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 09:38:58 2022

@author: DARSHAN HEGDE
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report
import seaborn as sns
from sklearn import model_selection, preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


data=pd.read_csv("E:\world_risk_index12.csv")
print(data)


sh=data.shape
print(sh)

n1=sh[0]



#    printing the disasters and its count with for loop



print("\t\tTHE LIST OF ALL DISASTERS AND IT'S NUMBER OF OCCURANCE IN THE WHOLE WORLD")
print('-------------------------------------------------------------------------------------')


count=[]
dis=(data["Disaster"].unique())
print(dis)
for i in  dis:
    cnt=0
    for j in range(n1):
        rec=data.iloc[j]
        if(i==rec[12]):
            cnt=cnt+1
    count.append(cnt)
    
print(count)



x=['Flood','Cyclone','Tsunami','Earthquake','Cold Wave','Landslide','Lightning','Drought','Heat Wave','Strong Wind','Volcano','Wildfire','Ice Storm']   



plt.pie(count,labels=x,radius=2.5,autopct='%0.1f%%')
plt.show()



     #    printing the disasters and its count without for loop

# print("\t\tTHE LIST OF ALL DISASTERS AND IT'S NUMBER OF OCCURANCE IN THE WHOLE WORLD")
# print('-------------------------------------------------------------------------------------')
# a=data['Disaster'].value_counts()
# print(a)


# x=['Cold Wave','Heat Wave','Drought','Tsunami','Earthquake','Landslide','Lightning','Flood','Wildfire','Ice Storm','Volcano','Strong Wind','Cyclone']   
# y=np.array([333,247,173,153,146,141,139,138,110,107,106,97,27])


# plt.pie(y,labels=x,radius=2.3,autopct='%0.1f%%')
# plt.show()





print("-------------------------------------------------------------------------------------------------------------------------------------------------------")
reg=input("\t\tENTER THE REGION NAME TO FIND THE AVERAGE OF WRI,EXPOSURE,VULNERABILITY,SUSPECTIBILITY,LACK OF COPING CAPABILITIES,LACK OF ADAPTIVE CAPACITIES:\n")


data2=data[data.Region==reg]
n = data2[["WRI","Exposure","Vulnerability","Susceptibility","Lack of Coping Capabilities","Lack of Adaptive Capacities"]]




print(n)
sh1=data2.shape
n2=sh[0]

print(sh1)

n2=sh1[0]
print('-------------------------------------------------------------------------------------')
averageper=[]
for i in range(1,7):
    sum=0
    cnt=0
    for j in range(n2):
        rec=data.iloc[j]
        d=float(rec[i])
        sum=d+sum
        cnt=cnt+1
    
    avg=sum/cnt
    averageper.append(avg)
print(averageper)


print('-------------------------------------------------------------------------------------')

x=["WRI","Exposure","Vulnerability","Susceptibility","Coping 0Capabilities","Adaptive Capacities"]



#function to plot

plt.title("bar chart")
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.bar(x,averageper,color=['yellow', 'red', 'green', 'blue', 'cyan','brown'])
plt.xticks(x,rotation=90)
plt.show()

print('\n')



print('\n')
print('----------------------------------------------------------------------------')
print("CHOOSE ANY ONE OPTION")
print("\t1:\tREGION WISE INFORMATION\n\t2:\tYEAR WISE INFORMATION\n\t3:\tDISASTER WISE INFORMATION\n\t4:\tREGION AND YEAR WISE INFORMATION\n")



opt=int(input("\t\tENTER YOUR CHOICE:\t"))

print('\n')
if(opt==1):
    
    region=input("\t\tENTER THE REGION:\t")
   
    a=data[data.Region==region]
    print('-----------------------------------------------------------------------------------------------------')
    print(a)  
    print('-----------------------------------------------------------------------------------------------------')
    
elif(opt==2):
    
    year=int(input("\t\tENTER THE YEAR FROM 2011 TO 2021:\t"))
   
    print("option 2 working")
    n=data[data.Year==year]
    print('-----------------------------------------------------------------------------------------------------')
    print(n)
    print('-----------------------------------------------------------------------------------------------------')
   
elif(opt==3):
   
    disaster=input("\t\tENTER THE DISASTER TYPE:\t")
    
    n=data[data.Disaster==disaster]
    print('-----------------------------------------------------------------------------------------------------')
    print(n)
    print('-----------------------------------------------------------------------------------------------------')
    
elif(opt==4):
    
    print("TO FIND REGION AND YEAR WISE INFORMATION")
    n=input("ENTER THE REGION:")
    print("----------------------------------")
    print("ENTER THE YEAR FROM 2011 TO 2021\n")
    year=int(input("enter the year:"))
    data1=data[data.Year==year]
    regdata=data1[data1.Region==n]
    print("---------------------------------------------------------------------------------------")
    print(regdata)
    print("---------------------------------------------------------------------------------------")
    
else:
    print("\t\t\tNO DISASTER FOUND IN A PARTICULAR YEAR!!!!!!!!!!")
    print('-----------------------------------------------------------------------------------------------------')
    

    
    

    

k=int(input("\t\tENTER THE YEAR TO FIND TOP 10 AND BOTTOM 10 WORLD RISK INDEX OF THE YEAR:\t"))
print('-----------------------------------------------------------------------------------------------------')
print("\n")

data2=data[data.Year==k]
b=data2.nlargest(10,['WRI'])

a=b.loc[:,["WRI","Region","Year","Disaster"]]

print("\t\tThe top 10 regions with highest World Risk Index are:")
print('----------------------------------------------------------------------------')
print(a)
print('----------------------------------------------------------------------------')
print("\n")

data2=data[data.Year==k]
b=data2.nsmallest(10,['WRI'])

a=b.loc[:,["WRI","Region","Year","Disaster"]]

print("\t\tTHE TOP 10 REGIONS WITH LOWEST WORLD RISK INDEX ARE:")

print('----------------------------------------------------------------------------')
print(a)
print('----------------------------------------------------------------------------')




data3=pd.read_csv("E:\WorldRiskIndex4.csv")
print(data3)
n3=data3.shape;

print(n3)
print(n3[0])
print(data3)

def convert(data3):
    number  = preprocessing.LabelEncoder()
    data3['Region'] = number.fit_transform(data3.Region)
    data3['Year'] = number.fit_transform(data3.Year)
    data3['Exposure_Category'] = number.fit_transform(data3.Exposure_Category)
    data3['WRI_Category'] = number.fit_transform(data3.WRI_Category)
    data3['Vulnerability_Category'] = number.fit_transform(data3.Vulnerability_Category)
    data3['Susceptibility_Category'] = number.fit_transform(data3.Susceptibility_Category)
    return data3
print()
print(convert(data3))
X =convert(data3.iloc[:,0:6])
print(X)

y=data3.iloc[:,-1]
print(y)

print("\ntotal number of x values :",X.shape)
print("total number of y values :",y.shape)

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.1,random_state=0)
print("\n x_train size \t x_test size")
print(X_train.shape,"\t\t",X_test.shape)
print("\n y_train size \t y_test size")
print(y_train.shape,"\t\t\t",y_test.shape)

print(y_test.shape)
knn = KNeighborsClassifier(n_neighbors=3,metric='euclidean') ## creation of model
print(knn.fit(X_train, y_train))
print(knn)
y_predict1=knn.predict(X_test)
#print(y_predict1)

#print("\nAccuracy =",knn.score(X_test, y_test))
acc=accuracy_score(y_test,y_predict1)
print("\nAccuracy =",acc)
cm=confusion_matrix(y_test.values,y_predict1)
print(cm)
df_cm = pd.DataFrame(cm, columns=np.unique(y_test), index = np.unique(y_test))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (10,7))
sns.set(font_scale=1.4)#for label size
sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})








