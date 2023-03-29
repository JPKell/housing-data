import os
# Third party imports
from flask import url_for
# Local imports
import model 
import style

def html() -> str:
    ''' Returns the HTML for the page. '''

    # Load a cached version of the page if it exists
    # if os.path.exists('cache/data-preparation.html'):
    #     with open('cache/data-preparation.html', 'r') as f:
    #         return f.read()

    df = model.get_df_from_csv('data/housing.csv')

    # Remove postsale features
    df = model.drop_post_sale_columns(df)

    # Remove Id column
    df = model.drop_columns(df, ['Id'])

    # Impute missing values
    nulls = model.get_nulls(df)

    # Create column for percentage of values missing
    null_percents = nulls / len(df)
    acceptable_nulls = 0.33 # We will drop columns with more than 33% missing values
    null_percents = null_percents[null_percents > acceptable_nulls]
    null_percents_table = [f'<tr><td>{name}</td><td>{round(percent * 100, 2)}%</td></tr>' for name, percent in null_percents.items() ]
    unacceptable_nulls = [name for name, percent in null_percents.items() if percent > acceptable_nulls]

    # Drop columns with more than the acceptable percentage of missing values
    df = model.drop_columns(df, unacceptable_nulls)

    # Impute missing values
    impute_zeros =['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'GarageYrBlt', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',]
    df = model.impute_zeros(df, impute_zeros)

    # Get the nulls again withouth the columns that were dropped
    nulls = model.get_nulls(df)
    null_col = df[[name for name in nulls.keys()]]
    
    df, impute_report = model.impute_missing_values(df, nulls.keys())
    imput_report_table = [f'<tr><td>{name}</td><td>{ value }</td></tr>' for name, value in impute_report.items() ]

    # Feature engineering
    df = model.create_total_bathrooms(df)
    df = model.create_total_square_feet(df)
    df = model.create_total_porch_square_feet(df)
    df = model.create_total_square_feet_per_room(df)
    features_removed = ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
    features_consolidated = ['TotalBathrooms', 'TotalSquareFeet', 'TotalPorchSquareFeet', 'TotalSquareFeetPerRoom', 'TotalQuality']

    # Create dummy variables
    categorical_keys = model.get_non_numeric_columns(df)
    
    drop_dominant_cols = ['Street', 'Utilities', 'Condition2', 'Heating', 'RoofMatl', 'BsmtFinType2', 'BsmtCond']
    poor_categories = []
    active = 'active'
    for key in drop_dominant_cols:
        html_str = f'''
            <div class="carousel-item {active}">
                <div class="d-flex justify-content-center">
                    { model.svg_categorical_bar(df[key], title=key) }
                </div>
            </div>
        '''
        active = ''
        poor_categories.append(html_str)

    # Drop columns that are not useful
    df = model.drop_columns(df, drop_dominant_cols)

    # Remove similar coloumns
    similar_keys = ['LandContour', 'LandSlope', 'Exterior1st', 'Exterior2nd', 'GarageQual', 'GarageCond']
    similar_cats = []
    for key in similar_keys:
        html_str = f'<div class="col-6 my-3 d-flex justify-content-center">{ model.svg_categorical_bar(df[key], title=key, plain=True) }</div>'
        similar_cats.append(html_str)
    drop_similar_keys = similar_keys[1::2]
    df = model.drop_columns(df, drop_similar_keys)

    bin_list = []
    # Binarize columns
    bin_keys = ['LandContour', 'Condition1', 'PavedDrive', 'CentralAir', 'Electrical', 'Functional', 'GarageQual']
    bin_list += bin_keys
    df = model.category_reduction(df, bin_keys, top=1)
    # 3
    bin_keys = ['RoofStyle', 'Foundation', 'MSZoning', 'LotShape']
    bin_list += bin_keys
    df = model.category_reduction(df, bin_keys, top=3)
    # 4
    bin_keys = ['LotConfig', 'BldgType']
    bin_list += bin_keys
    df = model.category_reduction(df, bin_keys, top=4)
    # 5
    bin_keys = ['HeatingQC', 'GarageType', 'HouseStyle']
    bin_list += bin_keys
    df = model.category_reduction(df, bin_keys, top=5)

    df = model.category_reduction(df, ['Exterior1st'], top=11 )
    bin_list += ['Exterior1st']
    categorical_keys = model.get_non_numeric_columns(df)
    reduced_categories = []
    active = 'active'
    for key in categorical_keys:
        html_str =  f'''           <div class="carousel-item {active}">
                <div class="d-flex justify-content-center">
                    { model.svg_categorical_bar(df[key], title=key) }
                </div>
            </div>'''
        active = ''
        reduced_categories.append(html_str)

    # Get the non-numeric columns again and create dummy variables
    categorical_keys = model.get_non_numeric_columns(df)
    size_before_dummies = len(df.keys())
    non_numerical_col_count = len(categorical_keys)
    df = model.create_dummy_columns(df)
    size_after_dummies = len(df.keys())

    # Automated feature selection algorithms
    temp_df = df.copy(deep=True)
    temp_df = model.drop_columns(temp_df, ['SalePrice']) 

    number_of_features = 20
    # Recursive feature elimination
    rfe = model.recursive_feature_selection(temp_df, df['SalePrice'], number_of_features)
    rfe.sort()

    # Forward feature selection
    ffe = model.forward_feature_selection(temp_df, df['SalePrice'], number_of_features)
    ffe.sort()

    # Feature importance
    feat_imp = model.feature_importance(temp_df, df['SalePrice'], number_of_features)
    feat_imp.sort()

    common_features = list(set(rfe).intersection(set(ffe)).intersection(set(feat_imp)))
    rfe_ffe_comm = list(set(rfe).intersection(set(ffe)))
    rfe_feat_comm = list(set(rfe).intersection(set(feat_imp)))
    ffe_feat_comm = list(set(ffe).intersection(set(feat_imp)))
    two_way_comm = list(set(rfe_ffe_comm + rfe_feat_comm + ffe_feat_comm))

    # Create tables
    rfe_table = ''
    for feature in rfe:
        highlight = ''
        if feature in common_features:
            highlight = style.top_hit
        elif feature in rfe_ffe_comm + rfe_feat_comm:
            highlight = style.secondary_hit

        rfe_table += f'<tr><td class="{highlight}">{feature}</td><td>'

    ffe_table = ''
    for feature in ffe:
        highlight = ''
        if feature in common_features:
            highlight = style.top_hit
        elif feature in rfe_ffe_comm + ffe_feat_comm:
            highlight = style.secondary_hit

        ffe_table += f'<tr><td class="{highlight}">{feature}</td><td>'


    feat_table = ''
    for feature in feat_imp:
        highlight = ''
        if feature in common_features:
            highlight = style.top_hit
        elif feature in rfe_feat_comm + ffe_feat_comm:
            highlight = style.secondary_hit

        feat_table += f'<tr><td class="{highlight}">{feature}</td><td>'

    # Multiple run results 
    _str1  = 'YearRemodAdd,  KitchenQual_Ex,  BsmtQual_Ex,  TotalBathrooms,  YearBuilt,  GarageArea,  GrLivArea,  OverallQual,  TotRmsAbvGrd,  OverallCond,  Fireplaces,  TotalSquareFeet,  GarageCars,  TotalBsmtSF '
    _str1 += 'GarageArea,  Fireplaces,  BsmtQual_Ex,  TotRmsAbvGrd,  YearBuilt,  TotalBsmtSF,  KitchenQual_Ex,  OverallCond,  GarageCars,  TotalBathrooms,  OverallQual,  TotalSquareFeet,  YearRemodAdd,  GrLivArea'
    _str1 += 'GarageArea,  Fireplaces,  BsmtQual_Ex,  TotRmsAbvGrd,  YearBuilt,  TotalBsmtSF,  KitchenQual_Ex,  OverallCond,  GarageCars,  TotalBathrooms,  OverallQual,  TotalSquareFeet,  TotalSquareFeetPerRoom,  GrLivArea,  YearRemodAdd'
    _str1 += 'GarageArea,  Fireplaces,  BsmtQual_Ex,  TotRmsAbvGrd,  YearBuilt,  TotalBsmtSF,  KitchenQual_Ex,  OverallCond,  1stFlrSF,  GarageCars,  TotalBathrooms,  OverallQual,  TotalSquareFeet,  YearRemodAdd,  GrLivArea'
    _str1 += 'GarageArea,  Fireplaces,  BsmtQual_Ex,  TotRmsAbvGrd,  YearBuilt,  TotalBsmtSF,  KitchenQual_Ex,  OverallCond,  GarageCars,  TotalBathrooms,  OverallQual,  TotalSquareFeet,  TotalSquareFeetPerRoom,  GrLivArea,  YearRemodAdd'
    _str1 += 'YearRemodAdd,  TotalSquareFeetPerRoom,  OverallQual,  TotalBsmtSF,  TotRmsAbvGrd,  KitchenQual_Ex,  BsmtQual_Ex,  TotalSquareFeet,  GarageCars,  OverallCond,  TotalBathrooms,  GrLivArea,  YearBuilt,  GarageArea,  Fireplaces'
    _str1 += 'TotalBsmtSF,  KitchenQual_Ex,  TotalBathrooms,  YearBuilt,  YearRemodAdd,  OverallQual,  TotalSquareFeet,  TotRmsAbvGrd,  GarageCars,  OverallCond,  BsmtQual_Ex,  GrLivArea,  Fireplaces,  GarageArea'

    _feat_dict = {}
    _features = _str1.split(',')
    _features = [x.strip() for x in _features]
    _features = [x for x in _features if x != '']

    for _feat in _features:
        if _feat not in _feat_dict:
            _feat_dict[_feat] = 1
        else:
            _feat_dict[_feat] += 1

    _feat_dict = {k: v for k, v in sorted(_feat_dict.items(), key=lambda item: item[1], reverse=True)}

    final_table = ''
    for _feat in _feat_dict:
        final_table += f'<tr><td>{_feat}</td><td>{_feat_dict[_feat]}</td></tr>'

    final_column_selection = [x for x in _feat_dict.keys() if _feat_dict[x] >= 5]

    # Create dataframes of the columns we are keeping and those we are dropping
    df_keep = df[final_column_selection].copy(deep=True)
    df_drop = df[[x for x in df.columns if x not in final_column_selection]].copy(deep=True)
    target = df['SalePrice']

    # pickle the dataframes
    df_keep.to_pickle('data/df_keep.pkl')
    df_drop.to_pickle('data/df_drop.pkl')
    target.to_pickle('data/target.pkl')

    with open('data_preparation.py', 'r') as f:
        code = f.read().replace('<','&lt;').replace('>','&gt;')
    html_str = f'''
<div class="row mt-5" style="height:300px;">
    <img src="{url_for('static', filename='banner-home.jpg')}" alt="Homes" style="width:100%;height:300px;object-fit: cover;">
</div>    
<div class="row mt-5">
    <h1>Data preparation</h1>

    <p> In this section we will look at the data and how it was prepared for modeling. Using knowledge gained from the initial
        analysis we will transform the data to improve the accuracy of the model. Some of the issues that require addressing are:
    </p>
    <div class="col">
        <ul>
            <li>Impute missing values</li>
            <li>Initial feature reduction</li>
            <li>Create dummy variables</li>
            <li>Bin variables</li>
            <li>Feature reduction through feature engineering</li>
            <li>Automated feature selection algorithms</li>
        </ul>
    </div>
    <p> The first thing to do is drop any post-sale data collected. We will not have this during production so lets get rid of it now. </p>
</div>
<hr class="my-5" />
<div class="row">
    <div class="col">
        <h2>Impute missing values</h2>
        <p class="mt-3"> The first step in preparing the data for modeling is to impute the missing values. There are {len(unacceptable_nulls)} columns with 
            greater than { acceptable_nulls * 100 }% null values. These columns will be discarded as they wont be as easy to gather 
            when feeding the final model production data. 
        </p>

        <p> A number of missing values are more likely to be features missing from the home rather than missing values. Therefore basement 
            and garage features should be set to 0 or 'None' if they have nulls</p>
        <p>Of the remaining columns with null values, there are a few numeric columns which are mainly integer values, finding the meadian will be 
           better than the mean here. For categorical columns we will opt for the largest mode. </p>
    </div>
    <div class="col-6">
        <table class='{ style.table } my-5'>
            <tr><th>Dropped</th><th>Percent missing</th></tr>
            { ''.join(null_percents_table) }
        </table>
        <table class='{ style.table }'>
            <tr><th>Feature</th><th>Value imputed</th></tr>
            { ''.join(imput_report_table) }
        </table>
    </div>
    <hr class="my-5" />
    <div class="col">
        <h2>Initial feature reduction</h2>
        <h3 class="mt-5">Feature engineering</h3>
        <p class="mt-3"> The next step is to reduce the number of features through feature engineering. There are numerous features
            that can be combined to create new features. This will reduce the number of features in the model, and give an easier value 
            to input to the final model. A total of { len(features_removed) } features were removed and { len(features_consolidated) } 
            features were created.</p>
        <p><strong>Features created:</strong>&nbsp;{', '.join(features_consolidated)}</p>

        <h3 class="mt-5">Remove columns with dominant categories</h3>
        <p> Plotting the categorical columns shows that there are { len(drop_dominant_cols) } of columns that have a single dominant value. These columns
            will be dropped as they will not add any value to the model. </p>
        <div class="row">
            <div id="deadFeatures" class="carousel carousel-dark carousel-fade slide" data-bs-ride="carousel">
                <div class="carousel-inner">
                    { ''.join(poor_categories) }
                </div>
                <button class="carousel-control-prev" type="button" data-bs-target="#deadFeatures" data-bs-slide="prev">
                    <span class="carousel-control-prev-icon" aria-hidden="true"></span>
                    <span class="visually-hidden">Previous</span>
                </button>
                <button class="carousel-control-next" type="button" data-bs-target="#deadFeatures" data-bs-slide="next">
                    <span class="carousel-control-next-icon" aria-hidden="true"></span>
                    <span class="visually-hidden">Next</span>
                </button>
            </div>
        </div>
        <p> There are similar columns that have similar distributions. Only one column per pair will remain</p>

        <p><strong>Features removed:</strong>&nbsp;{', '.join(drop_similar_keys)}</p>

        <div class="{ style.accordion } mb-5" id="dfDetails">
            <div class="accordion-item">
                <h2 class="accordion-header" id="headingOne">
                    <button class="{ style.accordion_button }" type="button" data-bs-toggle="collapse" data-bs-target="#collapseData">
                        Similar feature bar charts
                    </button>
                </h2>
                <div id="collapseData" class="accordion-collapse collapse" data-bs-parent="#dfDetails">
                    <div class="accordion-body">
                        <div class="row">
                            { ''.join(similar_cats) }
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <p>
            The dummy variables will create a large number of columns and to make them as effective as possible I 
            consolidated sparse categorical data with binning columns. In some cases they are binarized, others I 
            take the top categories and group the rest. A total of { len(bin_list) } columns were binned.
        </p>            
        <p><strong>Features binned:</strong>&nbsp;{', '.join(bin_list)}</p> 
    </div>
    <hr class="my-5" />
    <div class="col">
        <h2>Create dummy variables</h2>
        <p class="mt-3"> The next step is to create dummy variables for the categorical columns. There are { size_before_dummies}
            columns before creating the dummy variables, of which { non_numerical_col_count } are categorical.
            This caused the dataset to balloon to { size_after_dummies } columns after the process. 
            This is an increase of { size_after_dummies - size_before_dummies } columns.
            This can be reduced up front be creating some binary bins for some of the categorical columns. 
        </p>
    </div>
    <hr class="my-5" />
    <div class="col">
        <h2>Automated feature selection algorithms</h2>
        <p class="mt-3"> 
            With so many features it's difficult to know which ones are important and which ones are not. 
            You do not want to start guessing at what features are important, I will use a few automated feature selection algorithms
            to help me decide which features to keep. 
        </p>
        <p> Out of { number_of_features } features, there were only { len(common_features) } common features between all 
            three sets of selected variables. { len( two_way_comm ) } features were common between two of the three sets.
        </p>
        <p><strong>Common features</strong>: { ',&nbsp; '.join(two_way_comm) }.
        <p/>
        <div class="row mt-5">
            <div class="col-4 pe-1">
                <div class="card h-100 text-center">
                    <div class="card-header {style.bs_card_header}">Recursive feature elimination</div>
                    <div class="card-body">
                        RFE is a feature selection algorithm that works by recursively removing 
                        attributes and building a model on those attributes that remain.
                    </div>
                    <table class="{ style.table }">
                        { rfe_table }
                    </table>
                </div>
            </div>
            <div class="col-4 pe-1">
                <div class="card h-100 text-center">
                    <div class="card-header {style.bs_card_header}">Forward feature selection</div>
                    <p class="mt-3">
                        Forward feature selection is an algorithm that works by finding the best feature, and then adding the next. The model 
                        is built on next best feature, and so on.
                    </p>
                    <table class="{ style.table }">
                        { ffe_table }
                    </table>
                </div>
            </div>
            <div class="col-4">
                <div class="card h-100 text-center">
                    <div class="card-header {style.bs_card_header}">Feature importance</div>
                    <p class="mt-3">
                        Feature importance will be calculated based on the model coefficients. This will give an indication of which features
                        are most important to the model.
                    </p> 
                    <span class="align-bottom">                  
                        <table class="{ style.table } align-bottom">
                            { feat_table }
                        </table>
                    </span>
                </div>
            </div>
        </div>
        <p class="mt-5"> After running this a number of times and tracking the results. The features I will go into model building are </p>
        <p><strong>Features selected:</strong>&nbsp; { ', &nbsp; '.join(final_column_selection) }</p>
    </div>
    <hr class="my-5" />
    <div class="row">
        <p>Lets continue on to <a href="/data-preparation">creating the model</a></p>
    </div>
</div>
<hr class="my-5" />
<div class="row">
    <div class="{ style.accordion } mb-5" id="code">
        <div class="accordion-item">
            <h2 class="accordion-header" id="headingOne">
                <button class="{ style.accordion_button }" type="button" data-bs-toggle="collapse" data-bs-target="#collapseData">
                   Page code
                </button>
            </h2>
            <div id="collapseData" class="accordion-collapse collapse" data-bs-parent="#code">
                <div class="accordion-body">
                    <div class="row overflow-auto">
                        <pre>{ code }</pre>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

    '''

    # Cache the page for faster loading 
    with open('cache/data-preparation', 'w') as f:
        f.write(html_str)

    return html_str



