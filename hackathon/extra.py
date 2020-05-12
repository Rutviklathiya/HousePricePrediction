import numpy as np
from flask import Flask, request, jsonify, render_template
import random
import pickle
import pandas as pd



test = pd.read_csv("df_test_final.csv")
rand_idx = random.randrange(1000) 
final_fea = test.iloc[:,1:]
app = Flask(__name__)
model = pickle.load(open('final_model.pkl', 'rb'))
data=[
{
    'Utilities':None,
    'HouseStyle': None,
    'Exterior1st':None,
}]

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/get',methods=['GET'])
def get():
    return render_template('get.html')

@app.route('/blmngtn')
def blmngtn():
    return render_template('Blmngtn.html')

@app.route('/gilbert')
def gilbert():
    return render_template('Gilbert.html')

@app.route('/northridgeheights')
def northridgeheights():
    return render_template('NoRidge.html')

@app.route('/sawyer')
def sawyer():
    return render_template('Sawyer.html')

@app.route('/veenker')
def veenker():
    return render_template('Veenker.html')

@app.route('/brookside')
def brookside():
    return render_template('BrkSide.html')

@app.route('/edwardschool')
def edwardschool():
    return render_template('Edwards.html')

@app.route('/bluestem')
def bluestem():
    return render_template('Blueste.html')

@app.route('/crawford')
def crawford():
    return render_template('Crawfor.html')

@app.route('/collegecreek')
def collegecreek():
    return render_template('CollgCr.html')

@app.route('/meadaw')
def meadaw():
    return render_template('MeadowV.html')

@app.route('/mitchell')
def Mitchel():
    return render_template('Mitchel.html')


@app.route('/blmngtn_redirection')
def reBlmngtn():
    return render_template('Blmngtn_redirection.html')

@app.route('/blueste_redirection')
def reBlueste():
    return render_template('Blueste_redirection.html')

@app.route('/brkside_redirection')
def reBrkside():
    return render_template('BrkSide_redirection.html')

@app.route('/collgcr_redirection')
def recollgcr():
    return render_template('Collgcr_redirection.html')

@app.route('/crawfor_redirection')
def recrawfor():
    return render_template('Crawfor_redirection.html')

@app.route('/edwards_redirection')
def reedwards():
    return render_template('Edwards_redirection.html')

@app.route('/meadowv_redirection')
def remeadowv():
    return render_template('MeadowV_redirection.html')

@app.route('/mitchel_redirection')
def remitchel():
    return render_template('Mitchel_redirection.html')

@app.route('/noridge_redirection')
def renoridge():
    return render_template('Noridge_redirection.html')

@app.route('/sawyer_redirection')
def resawyer():
    return render_template('Sawyer_redirection.html')

@app.route('/veenker_redirection')
def reveenker():
    return render_template('Veenker_redirection.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    print(int_features)
    global data
    data=[
    {
        'Utilities':int_features[0],
        'HouseStyle': int_features[2],
        'Exterior1st':int_features[3],
    }]
   


    
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('formulatedtest.csv')
    clm = ['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
           'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
           'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
           'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
           'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
           'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
           'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
           'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
           'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
           'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
           'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
           'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
           'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
           'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
           'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
           'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
           'SaleCondition']
    row = [1465, 120, 'RL' , 43 , 5005,    'Pave' ,   'NA',  'IR1', 'HLS','AllPub',  'Inside',  'Gtl','StoneBr', 'Norm',    'Norm',    'TwnhsE',  '1Story',  8,   5,   1992 ,   1992,    'Gable',   'CompShg', 'HdBoard', 'HdBoard' ,None ,   0 ,  'Gd',  'TA',  'PConc',   'Gd',  'TA' , 'No',  'ALQ',263 ,'Unf', 0 ,  1017,    1280 ,   'GasA'    ,'Ex'  ,'Y'   ,'SBrkr'   ,1280    ,0   ,0   ,1280    ,0   ,0   ,2   ,0  , 2  , 1   ,'Gd'  ,5   ,'Typ' ,0   ,'NA'  ,'Attchd'  ,1992    ,'RFn',2   ,506 ,'TA'  ,'TA'  ,'Y'  ,0   ,82  ,0   ,0   ,144 ,0   ,'NA'  ,'NA'  ,'NA'  ,0   ,1   ,2010    ,'WD'  ,'Normal']

    row = np.asarray(row).reshape(1,80)
    new_row = pd.DataFrame(row,columns=clm)

    new_row['Utilities'] = int_features[0]
    new_row['Neighborhood'] = int_features[1]
    new_row['HouseStyle'] = int_features[2]
    new_row['Exterior1st'] = int_features[3]
    new_row['1stFlrSF'] = int_features[4]
    new_row['2ndFlrSF'] = int_features[5]
    new_row['GrLivArea'] = int_features[6]
    new_row['FullBath'] = int_features[7]
    new_row['BedroomAbvGr'] = int_features[8]

    df_train = df_train.iloc[:,:80]
    df=pd.concat([df_train,df_test],axis=0)
    df=pd.concat([df,new_row],axis=0)
    df['LotFrontage']=df['LotFrontage'].fillna(df['LotFrontage'].mean())
    df.drop(['Alley'],axis=1,inplace=True)
    df['BsmtCond']=df['BsmtCond'].fillna(df['BsmtCond'].mode()[0])
    df['BsmtQual']=df['BsmtQual'].fillna(df['BsmtQual'].mode()[0])
    df['FireplaceQu']=df['FireplaceQu'].fillna(df['FireplaceQu'].mode()[0])
    df['GarageType']=df['GarageType'].fillna(df['GarageType'].mode()[0])
    df.drop(['GarageYrBlt'],axis=1,inplace=True)
    df['GarageFinish']=df['GarageFinish'].fillna(df['GarageFinish'].mode()[0])
    df['GarageQual']=df['GarageQual'].fillna(df['GarageQual'].mode()[0])
    df['GarageCond']=df['GarageCond'].fillna(df['GarageCond'].mode()[0])
    df.drop(['PoolQC','Fence','MiscFeature'],axis=1,inplace=True)
    df.drop(['Id'],axis=1,inplace=True)
    #df.drop(['Blmngtn'],axis=1,inplace=True)
    df['MasVnrType']=df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])
    df['MasVnrArea']=df['MasVnrArea'].fillna(df['MasVnrArea'].mode()[0])
    df['BsmtExposure']=df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0])
    df['BsmtFinType2']=df['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0])
    df.dropna(inplace=True)
    columns=['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood',
             'Condition2','BldgType','Condition1','HouseStyle','SaleType',
            'SaleCondition','ExterCond',
             'ExterQual','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
            'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Heating','HeatingQC',
             'CentralAir',
             'Electrical','KitchenQual','Functional',
             'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive']

    def category_onehot_multcols(multcolumns):
        df_final=df
        i=0
        for fields in multcolumns:
            
            print(fields)
            df1=pd.get_dummies(df[fields],drop_first=True)
            
            df.drop([fields],axis=1,inplace=True)
            if i==0:
                df_final=df1.copy()
            else:
                
                df_final=pd.concat([df_final,df1],axis=1)
            i=i+1
           
            
        df_final=pd.concat([df,df_final],axis=1)
            
        return df_final

    final_df=category_onehot_multcols(columns)
    final_df =final_df.loc[:,~final_df.columns.duplicated()]
    final_df.drop('NA', axis=1, inplace=True)
    change_clm = ['1stFlrSF', '2ndFlrSF', '3SsnPorch', 'BedroomAbvGr', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtUnfSF', 'EnclosedPorch', 'Fireplaces', 'FullBath', 'GarageArea', 'GarageCars', 'GrLivArea', 'HalfBath', 'KitchenAbvGr', 'LotArea', 'LowQualFinSF', 'MSSubClass', 'MiscVal', 'MoSold', 'OpenPorchSF', 'OverallCond', 'OverallQual', 'PoolArea', 'ScreenPorch', 'TotRmsAbvGrd', 'TotalBsmtSF', 'WoodDeckSF', 'YearBuilt', 'YearRemodAdd', 'YrSold']
    for clms in change_clm:
        final_df[clms]=final_df[clms].astype(int)




    prediction = model.predict(final_df)
    # output = round(prediction[0], 2)
    # rand_idx = random.randrange(1000) 

    return render_template('index.html', prediction_text='Price of the house should be ${}'.format(prediction[-1]))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


@app.route('/gilbert_redirection')
def reGilbert():
    return render_template('gilbert_redirection.html',data=data)


if __name__ == "__main__":
    app.run(debug=True)