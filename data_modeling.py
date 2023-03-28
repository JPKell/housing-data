import os
# Third party imports
from flask import url_for
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model  import Lasso, LinearRegression, Ridge
from sklearn.tree          import DecisionTreeRegressor
from sklearn.ensemble      import RandomForestRegressor, GradientBoostingRegressor
# Local imports
import model
import style

def html() -> str:
    ''' Returns the HTML for the page. '''

    # # Load a cached version of the page if it exists
    # if os.path.exists('cache/data-modeling.html'):
    #     with open('cache/data-modeling.html', 'r') as f:
    #         return f.read()

    # Load the pickled data. There may be issues if the pages are run out of order and the model is not yet cached.
    # perhapts we keep the pkl files around when we flush the cache?
    train_df = pd.read_pickle('data/df_keep.pkl')
    target = pd.read_pickle('data/target.pkl')
    dropped = pd.read_pickle('data/df_drop.pkl') # Might not need or want this. 

    # Models
    # Linear Regression
    linear = LinearRegression()
    # Ridge Regression
    ridge = Ridge()
    # Lasso Regression
    lasso = Lasso()
    # Decision Tree
    dec_tree = DecisionTreeRegressor()
    # Random Forest
    rand_forest = RandomForestRegressor(n_estimators=20)
    # Gradient Boosting
    grad_boost = GradientBoostingRegressor(n_estimators=20)

    model_list = [linear, ridge, lasso, dec_tree, rand_forest, grad_boost]
    
    model_list_cards = ''
    for mod in model_list:
        model_list_cards += f'''<div class="col-6 p-1"><div class="card h-100 text-center"><div class="card-body">{ mod }</div></div></div>'''
    # Scaling
    scaler_list = [StandardScaler, MinMaxScaler, RobustScaler, None]

    model_summary = {}
    folds = 5
    for mod in model_list:
        model_summary[mod.__class__.__name__] = {}
        for scaler in scaler_list:
            if scaler != None:
                model_summary[mod.__class__.__name__][scaler.__name__] = []
            else:
                model_summary[mod.__class__.__name__]['None'] = []

            result = model.evaluate_model(mod, scaler, train_df, target, folds=folds)

            for d in result:
                if scaler != None:
                    model_summary[mod.__class__.__name__][scaler.__name__].append(d['RMSE'])
                else:
                    model_summary[mod.__class__.__name__]['None'].append(d['RMSE'])

    # Gather stats for the table so we can highlight the best results
    stats = {}
    for m in model_summary:
        for s in model_summary[m]:
            stats[s] = {'mean':[], 'std':[]}
        break

    for _model in model_summary:
        for _scaler, value in model_summary[_model].items():
            mean = np.mean(value)
            std  = np.std(value)
            stats[_scaler]['mean'].append(mean)
            stats[_scaler]['std'].append(std)

    # Crossfold Validation table
    folds_table = ''
    for m in model_summary:
        folds_table += f'<tr><td>{ m }</td>'
        for scaler, data in model_summary[m].items():
            mean = np.mean(data)
            std  = np.std(data)

            is_min_mean = mean == min(stats[scaler]['mean'])
            is_min_std  = std == min(stats[scaler]['std'])
        
            mean_hilight = style.table_highlight if is_min_mean else ''
            std_hilight  = style.table_highlight if is_min_std else ''

            folds_table += f'<td class="{mean_hilight}">{mean:.2f}</td><td class="{std_hilight}">{std:.2f}</td>'
                
        folds_table += '</tr>'

    # Stacking

    # Test train split
    X_train, X_test, y_train, y_test = model.train_test_split(train_df, target, test_size=0.2)
    stacked_model = model.stack_models(model_list, X_train, y_train)

    # predict the test data
    y_pred = stacked_model.predict(X_test)

    # Calculate the RMSE
    result = model.evaluate_model(stacked_model, None, train_df, target, folds=folds)
    stacked_table = ''
    print(result)
    # stacked_table += f'<tr><td>Stacked model</td>'
    # for scaler, data in result.items():
    #     mean = np.mean(data)
    #     std  = np.std(data)

    #     is_min_mean = mean == min(stats[scaler]['mean'])
    #     is_min_std  = std == min(stats[scaler]['std'])
    
    #     mean_hilight = style.table_highlight if is_min_mean else ''
    #     std_hilight  = style.table_highlight if is_min_std else ''

    #     stacked_table += f'<td class="{mean_hilight}">{mean:.2f}</td><td class="{std_hilight}">{std:.2f}</td>'
            
    # stacked_table += '</tr>'

    html_str = f'''
<div class="row mt-5" style="height:300px;">
    <img src="{url_for('static', filename='banner-home.jpg')}" alt="Homes" style="width:100%;height:300px;object-fit: cover;">
</div>    
<h1 class="mt-5">Build the data model</h1>
<p> 
    There are multiple models to be built and evaluated. As noted in the initial exploration the data skews right and 
    for many of the models that will bias the results. Scaling should help to normalize the data. Models must be selected
    and evaluated. Then tweaked and evaluated again.
</p>
<hr class="my-5" />
<h2>Scaling</h2>
<p> 
    The data is skewed right. Scaling should help to normalize the data. During model building I evaluated { len(scaler_list) }
    different scaling methods. The results were .... 
</p>
<p><strong>Scalers used:</strong> {', '.join([x.__name__ for x in scaler_list if not isinstance(x, type(None))])}</p>
<hr class="my-5" />
<h2>Models</h2>
<p> 
    The data is a linear regression problem. I evaluated the following models:
</p>
<div class="row">
    { model_list_cards }
</div> 
<hr class="my-5" />
<h2>Cross fold validation </h2>
<p> 
    The results of many runs are evaluated to determine the best model. The results of a single pass
    with { folds } folds are as follows:
</p>
<table class="{ style.table }">
    <thead>
        <tr>
            <th scope="col">Model</th>
            <th scope="col" style="width:10%;">StandardScaler</th>
            <th scope="col" style="width:10%;">Std dev.</th>
            <th scope="col" style="width:10%;">MinMaxScaler</th>
            <th scope="col" style="width:10%;">Std dev.</th>
            <th scope="col" style="width:10%;">RobustScaler</th>
            <th scope="col" style="width:10%;">Std dev.</th>
            <th scope="col" style="width:10%;">None</th>
            <th scope="col" style="width:10%;">Std dev.</th>
        </tr>
    </thead>
    <tbody>
        { folds_table }
    </tbody>
</table>
<p> Over a number of runs it was clear that the best model was the <strong>Random Forest</strong> model. </p>


<hr class="my-5" />
<h2>Stacking </h2>
<p> 
    Stacking is a method of combining the results of multiple models. The stacking model is built using the results of the
    best models from the cross fold validation. The results of the stacking model are as follows:
</p>

    '''

    with open('cache/data-modeling.html', 'w') as f:
        f.write(html_str)
    return html_str