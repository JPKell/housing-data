from io import StringIO
import subprocess

from matplotlib.backends.backend_svg import FigureCanvasSVG
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from sklearn.feature_selection  import f_regression, RFE
from sklearn.linear_model       import LinearRegression
from sklearn.model_selection    import train_test_split
from sklearn.preprocessing      import MinMaxScaler, RobustScaler, StandardScaler, MaxAbsScaler
from sklearn.tree               import DecisionTreeRegressor 
from sklearn.ensemble           import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model       import Ridge, Lasso, LinearRegression
from sklearn.model_selection    import GridSearchCV
from sklearn.pipeline           import make_pipeline
from sklearn.model_selection    import KFold, cross_val_score
from sklearn.metrics            import mean_squared_error
from sklearn.metrics            import r2_score

from sklearn.ensemble           import StackingRegressor

from   sklearn.linear_model import LinearRegression
from   sklearn.feature_selection import RFE

import style

# Read and write files
def get_df_from_csv(path='data/housing.csv') -> pd.DataFrame:
    ''' Returns a pandas dataframe from a csv file '''
    df = pd.read_csv(path)
    return df

def write_df_to_csv(df: pd.DataFrame, path='data/correlation.csv') -> None:
    ''' Writes a pandas dataframe to a csv file '''
    df.to_csv(path, index=False)

def get_correlation_results() -> str:
    ''' Returns the correlation results '''
    df = get_df_from_csv('data/corr_matrix.csv')
    df.rename(columns={'Unnamed: 0': 'Feature'}, inplace=True)
    return df


# R code 
def process_r_code(r_code: str) -> None:
    ''' Runs an R code script '''
    subprocess.run(['Rscript', r_code], stdout=subprocess.PIPE)

# Get data
def get_features_target(df: pd.DataFrame) -> tuple:
    ''' Returns a tuple of the features and target dataframes '''
    features = df.drop('SalePrice', axis=1)
    target = df['SalePrice']
    return features, target

def get_keys(df: pd.DataFrame) -> list:
    ''' Returns a list of the column names in a dataframe '''
    keys = list(df.keys())
    keys.sort()
    return keys

def get_nulls(df: pd.DataFrame) -> pd.DataFrame:
    nulls = df.isnull().sum().sort_values(ascending=False)
    nulls = nulls[nulls > 0]
    return nulls

def get_non_numeric_columns(df: pd.DataFrame) -> list:
    ''' Returns a list of the non numeric columns in a dataframe '''
    non_numeric = []
    for key in df.keys():
        if df[key].dtype != 'int64' and df[key].dtype != 'float64':
            non_numeric.append(key)
    return non_numeric

# Data cleaning
def drop_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    ''' Returns a dataframe with the specified columns dropped '''
    df = df.drop(columns, axis=1)
    return df

def drop_post_sale_columns(df: pd.DataFrame) -> pd.DataFrame:
    ''' Returns a dataframe with the post sale columns dropped '''
    post_sale_columns = ['MoSold', 'YrSold', 'SaleType', 'SaleCondition', 'Id']
    df = df.drop(post_sale_columns, axis=1)
    return df

def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    ''' Returns a dataframe with the numeric columns normalized '''
    df['SalePrice'] = (df['SalePrice'] - df['SalePrice'].mean()) / df['SalePrice'].std()
    for key in df.keys():
        if df[key].dtype == 'int64' or df[key].dtype == 'float64':
            df[key] = (df[key] - df[key].mean()) / df[key].std()
    return df

# Feature engineering
def impute_zeros(df: pd.DataFrame, columns:list) -> pd.DataFrame:
    ''' Returns a dataframe with the specified columns imputed with zeros '''
    for column in columns:
        if df[column].dtype == 'int64' or df[column].dtype == 'float64':
            df[column] = df[column].fillna(0)
        else:
            df[column] = df[column].fillna('None')
    return df

def impute_missing_values(df: pd.DataFrame, columns:list) -> tuple:
    ''' Returns a tuple of Dataframe and a dictionary of the imputed values '''
    report = {}
    for column in columns:
        # Check the column Dtype
        if df[column].dtype == 'int64' or df[column].dtype == 'float64':
            # Numeric columns primarily int's so use median
            median = df[column].median()
            df[column] = df[column].fillna(df[column].median())
            report[column] = median
        else:
            mode = df[column].mode()[0]
            df[column] = df[column].fillna(mode)
            report[column] = mode
    return df, report

def category_reduction(df: pd.DataFrame, columns:list, top:int) -> pd.DataFrame:
    ''' Returns a dataframe with the specified columns reduced to the top 10 most common values '''
    for column in columns:
        top_value = df[column].value_counts().index[:top-1]
        if top == 1:
            df[column] = df[column].apply(lambda x: 1 if x == top_value else 0)
        else:
            df[column] = df[column].apply(lambda x: x if x in top_value else 'Other')
    return df

def create_dummy_columns(df: pd.DataFrame) -> pd.DataFrame:
    ''' Returns a dataframe with dummy columns for non numeric columns '''
    non_numeric = get_non_numeric_columns(df)
    df = pd.get_dummies(df, columns=non_numeric)
    return df

def create_total_bathrooms(df: pd.DataFrame) -> pd.DataFrame:
    ''' Returns a dataframe with a total bathrooms column '''
    df['TotalBathrooms'] = df['FullBath'] + (0.5 * df['HalfBath']) + df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath'])
    return df

def create_total_square_feet(df: pd.DataFrame) -> pd.DataFrame:
    ''' Returns a dataframe with a total square feet column '''
    df['TotalSquareFeet'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    return df

def create_total_porch_square_feet(df: pd.DataFrame) -> pd.DataFrame:
    ''' Returns a dataframe with a total porch square feet column '''
    df['TotalPorchSquareFeet'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch']
    return df

def create_total_square_feet_per_room(df: pd.DataFrame) -> pd.DataFrame:
    ''' Returns a dataframe with a total square feet per room column '''
    df['TotalSquareFeetPerRoom'] = df['TotalSquareFeet'] / df['TotRmsAbvGrd']
    return df

def create_total_quality(df: pd.DataFrame) -> pd.DataFrame:
    ''' Returns a dataframe with a total quality column '''
    df['TotalQuality'] = df['OverallQual'] + df['OverallCond']
    return df

# Automated feature selection.

def forward_feature_selection(X,y,n_features, names=None):
    ffs = f_regression(X, y)
    if type(names) == type(None):
        names = X.columns
    ffs_feature_scores = list(zip(names,ffs[0]))
    ffs_feature_scores.sort(key=lambda x: x[1], reverse=True)
    return [ feat for feat, score in ffs_feature_scores[:n_features] ]

def recursive_feature_selection( X, y,n_features):
    model = DecisionTreeRegressor()
    rfe   = RFE(model, n_features_to_select=n_features)
    rfe   = rfe.fit(X, y)
    return [ name for i, name in enumerate(X.keys()) if rfe.support_[i] ]

def feature_importance(X,y,n_features, names=None):
    model = LinearRegression()
    model.fit(X,y)
    if type(names) == type(None):
        names = X.columns
    model_coef = list(zip(names,model.coef_))
    model_coef.sort(key=lambda x: x[1], reverse=True)
    return [ feat for feat, score in model_coef[:n_features] ]

# Evaluation
def evaluate_model(model, scaler, x, y, folds=5):
    ''' Returns a dictionary of the model evaluation metrics '''
   
    if not isinstance(scaler, type(None)):
        scaler = scaler()
    kf = KFold(n_splits=folds, shuffle=True)

    run_data = []
    for train_index, test_index in kf.split(x):
        if isinstance(scaler, type(None)):
            x_train = x.loc[train_index]
            x_test  = x.loc[test_index]
        else:
            x_train = scaler.fit_transform(x.loc[train_index])
            x_test  = scaler.transform(x.loc[test_index])

        y_train = y.loc[train_index]
        y_test  = y.loc[test_index]

        # Fit the model
        model.fit(x_train, y_train)

        # Predict the target
        y_pred = model.predict(x_test)

        # Calculate the evaluation metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        run_data.append({'Model': model, 'Scaler': scaler, 'RMSE': rmse})

    return run_data

def stack_models(models, x_train, y_train, x_test, y_test):
    ''' Returns a stacked model '''
    predictions = pd.DataFrame()

    for model in models:
        model.fit(x_train, y_train) 
        if isinstance(model, LinearRegression):
            predictions[model] = model.predict(x_test)
        predictions[model] = model.predict(x_test)

    stacked_model = LinearRegression()
    stacked_model.fit(predictions, y_test)

    return stacked_model
    
def model_stacker(models, x_train, y_train):
    ''' Returns a stacked model '''
    meta_model    = LinearRegression()
    stacked_model = StackingRegressor(estimators=models, final_estimator=meta_model)
    stacked_model.fit(x_train, y_train)
    return stacked_model



# Outputs
def svg_histogram(x,figsize=(10,5), x_lab:str=None, y_lab:str=None, title:str=None, **kwargs) -> str:
    ''' Creates a matplotlib histogram and returns the svg as a string '''
    fig = Figure(figsize=figsize, tight_layout=True)
    ax = fig.subplots(1,1)
    ax.hist(x, color=style.css_color,  rwidth=0.9,**kwargs)

    if x_lab:
        ax.set_xlabel(x_lab, **style.font)
    if y_lab:
        ax.set_ylabel(y_lab, **style.font)
    if title:
        ax.set_title(title, **style.font)

    # Rather than write out the file, just store it in a buffer. 
    with StringIO() as file:
        FigureCanvasSVG(fig).print_svg(file)
        svg = file.getvalue()

    svg = svg.replace('width="720pt" height="360pt"', 'width="100%" height="100%"')

    # Export the svg for the main page
    with open('static/salesPrice.svg', 'w') as f:
        f.write(svg)
    
    return svg

def svg_scatter(x, y, figsize=(10,5), x_lab:str=None, y_lab:str=None, title:str=None, **kwargs) -> str:
    ''' Creates a matplotlib scatter plot and returns the svg as a string '''
    fig = Figure(figsize=figsize, tight_layout=True)
    ax = fig.subplots(1,1)
    # scatter plot with regression line
    ax.scatter(x, y, c=style.css_color , **kwargs)
    ax.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), color=style.warning)

    if x_lab:
        ax.set_xlabel(x_lab, **style.font)
    if y_lab:
        ax.set_ylabel(y_lab, **style.font)
    if title:
        ax.set_title(title, **style.font)

    # Rather than write out the file, just store it in a buffer. 
    with StringIO() as file:
        FigureCanvasSVG(fig).print_svg(file)
        svg = file.getvalue()
    svg = svg.replace('width="720pt" height="360pt"', 'width="80%" height="100%"')

        # Export the svg for the main page
    with open(f'static/{title}.svg', 'w') as f:
        f.write(svg)

    return svg

def svg_categorical_bar(x:pd.DataFrame,plain=False, figsize=(10,5), x_lab:str=None, y_lab:str=None, title:str=None, **kwargs) -> str:
    ''' Creates a matplotlib categorical bar plot and returns the svg as a string '''
    fig = Figure(figsize=figsize, tight_layout=True)
    ax = fig.subplots(1,1)

    # get the unique values and their counts
    y = x.value_counts().values
    x = x.value_counts().index


    ax.bar(x, y, color=style.css_color, **kwargs)
    if plain:
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

    if x_lab:
        ax.set_xlabel(x_lab, **style.font)
    if y_lab:
        ax.set_ylabel(y_lab, **style.font)
    if title:
        new_font = style.font.copy()
        new_font['size'] = 26
        ax.set_title(title, **new_font)

    # Rather than write out the file, just store it in a buffer. 
    with StringIO() as file:
        FigureCanvasSVG(fig).print_svg(file)
        svg = file.getvalue()
    svg = svg.replace('width="720pt" height="360pt"', 'width="80%" height="100%"')
    return svg

def svg_rmse(df, figsize=(10,5), x_lab:str=None, y_lab:str=None, title:str=None, std_dev=False, legend=True, moving_average=10, **kwargs) -> str:
    ''' Creates a matplotlib line plot and returns the svg as a string '''
    fig = Figure(figsize=figsize, tight_layout=True)
    ax = fig.subplots(1,1)

    if std_dev:
        df = df.rolling(moving_average).std()
    else:
        # Create a moving average
        df = df.rolling(moving_average).mean()

    ax.plot(df,  **kwargs)

    if legend:
        ax.legend(df.columns)

    if x_lab:
        ax.set_xlabel(x_lab, **style.font)
    if y_lab:
        ax.set_ylabel(y_lab, **style.font)
    if title:
        ax.set_title(title, **style.font)

    # Rather than write out the file, just store it in a buffer. 
    with StringIO() as file:
        FigureCanvasSVG(fig).print_svg(file)
        svg = file.getvalue()
    svg = svg.replace('width="720pt" height="360pt"', 'width="100%" height="100%"')

    title = 'stdev' if std_dev else 'rmse'
    with open(f'static/{title}.svg', 'w') as f:
        f.write(svg)
    return svg