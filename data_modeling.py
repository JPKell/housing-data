import os, pickle
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

    # Load a cached version of the page if it exists
    if os.path.exists('cache/data-modeling.html'):
        with open('cache/data-modeling.html', 'r') as f:
            return f.read()

    # Load the pickled data. There may be issues if the pages are run out of order and the model is not yet cached.
    # perhapts we keep the pkl files around when we flush the cache?
    train_df = pd.read_pickle('data/df_keep.pkl')
    target = pd.read_pickle('data/target.pkl')

    # Models
    linear = LinearRegression()
    ridge  = Ridge(alpha=15.4)
    lasso  = Lasso(alpha=0.0006)
    dec_tree    = DecisionTreeRegressor(criterion='squared_error', max_depth=5)
    rand_forest = RandomForestRegressor(n_estimators=30, criterion='squared_error', max_depth=5)
    grad_boost  = GradientBoostingRegressor(n_estimators=100)

    # The tuple is for the stacker which needs a list of tuples
    model_list = [linear, ridge, lasso, dec_tree, rand_forest, grad_boost]
    
    model_list_cards = ''
    for mod in model_list:
        model_list_cards += f'''<div class="col-6 p-1"><div class="card h-100 text-center bg-{style.bs_color} text-white"><div class="card-body">{ mod }</div></div></div>'''
    # Scaling
    scaler_list = [StandardScaler, MinMaxScaler, RobustScaler, None]
    
    model_summary = {}
    folds = 5
    for mod in model_list:
        model_summary[mod.__class__.__name__] = {}
        for scaler in scaler_list:
            # Create a dictionary to store the results
            if scaler != None:
                model_summary[mod.__class__.__name__][scaler.__name__] = []
            else:
                model_summary[mod.__class__.__name__]['None'] = []

            result = model.evaluate_model(mod, scaler, train_df, target, folds=folds)

            # Store the results in the dictionary
            for d in result:
                if scaler != None:
                    model_summary[mod.__class__.__name__][scaler.__name__].append(d['RMSE'])
                else:
                    model_summary[mod.__class__.__name__]['None'].append(d['RMSE'])

    # Gather stats for the table so we can highlight the best results
    stats = {}
    for m in model_summary:
        for s in model_summary[m]:
            stats[s] = {'mean':[], 'sd':[]}
        break

    # Gather each scaler's mean and standard deviation for comparison
    for _model in model_summary:
        for _scaler, data in model_summary[_model].items():
            mean = np.mean(data)
            std  = np.std(data)
            stats[_scaler]['mean'].append(mean)
            stats[_scaler]['sd'].append(std)

    # Crossfold Validation table
    folds_table = ''
    for m in model_summary:
        folds_table += f'<tr><td>{ m }</td>'
        for scaler, data in model_summary[m].items():
            mean = np.mean(data)
            std  = np.std(data)

            is_min_mean = mean == min(stats[scaler]['mean'])
            is_min_std  = std == min(stats[scaler]['sd'])
        
            mean_hilight = style.table_highlight if is_min_mean else ''
            std_hilight  = style.table_highlight if is_min_std else ''

            folds_table += f'<td class="{mean_hilight}">{mean:,.2f}</td><td class="{std_hilight}">{std:,.2f}</td>'        
        folds_table += '</tr>'

    def evaluateModel(y_test, predictions, _model) -> float:
        mse = model.mean_squared_error(y_test, predictions)
        rmse = round(np.sqrt(mse),3)
        return rmse

    def fitBaseModels(X_train, y_train, X_test, models):
        dfPredictions = pd.DataFrame()

        # Fit base model and store its predictions in dataframe.
        for i in range(0, len(models)):
            models[i].fit(X_train, y_train)
            predictions = models[i].predict(X_test)
            colName = str(i)
            # Add base model predictions to column of data frame.
            dfPredictions[colName] = predictions
        return dfPredictions, models

    def fitStackedModel(X, y):
        model = LinearRegression()
        model.fit(X, y)
        return model

    # Split data into train, test and validation sets.
    X_train, X_temp, y_train, y_temp = model.train_test_split(train_df, target, test_size=0.70)
    X_test, X_val, y_test, y_val = model.train_test_split(X_temp, y_temp, test_size=0.50)

    # Fit base and stacked models.
    dfPredictions, models = fitBaseModels(X_train, y_train, X_test, model_list)
    stackedModel          = fitStackedModel(dfPredictions, y_test)

    # Evaluate base models with validation data.
    dfValidationPredictions = pd.DataFrame()
    for i in range(0, len(models)):
        predictions = models[i].predict(X_val)
        colName = str(i)
        dfValidationPredictions[colName] = predictions

    # Evaluate stacked model with validation data.
    stackedPredictions = stackedModel.predict(dfValidationPredictions)

    stacked_rmse = evaluateModel(y_val, stackedPredictions, stackedModel)

    stacked_results = [37588.89,  32568.18, 35342.92, 26740.74, 33926.85, 26698.88, 31879.72, 31576.92, 28993.93, 26088.86]
    stacked_mean = np.mean(stacked_results)

    # Save the model list for model evaluation next page.
    with open('data/model_list.pkl', 'wb') as f:
        pickle.dump(model_list,f)

    # Read the code for the page
    with open('data_modeling.py', 'r') as f:
        code = f.read().replace('<', '&lt;').replace('>', '&gt;')

    html_str = f'''
        <div class="row mt-5" style="height:300px;">
            <img src="{url_for('static', filename='banner-home.jpg')}" alt="Homes" style="width:100%;height:300px;object-fit: cover;">
        </div>    
        <h1 class="mt-5">Build the data model</h1>
        <p> 
            There are multiple models to be built and evaluated. A collection of regression algorithms will be used to build the model.
            Once the models are built, they will be evaluated using cross fold validation. Stacking will also be used to develop a 
            stacked model. The best model will be used to build to production model.
        </p>
        <hr class="my-5" />
        <h2>Models</h2>
        <p> 
            The data is a linear regression problem. I evaluated the following models:
        </p>
        <div class="row">
            { model_list_cards }
        </div> 
        <hr class="my-5" />
        <h2>Scaling</h2>
        <p> 
            The data is skewed right. Scaling should help to normalize the data. During model building I evaluated { len(scaler_list) }
            different scaling methods. 
        </p>
        <p><strong>Scalers used:</strong> {', '.join([x.__name__ for x in scaler_list if not isinstance(x, type(None))])}</p>
        <p>In the end none of the scalers improved the model.</p>
        <hr class="my-5" />
        <h2>Cross fold validation </h2>
        <p> 
            Root Mean Squared Error (RMSE) and Standard Deviation (SD) are both measures of the 
            spread of data. However, they are used in different contexts and have different 
            interpretations when evaluating a linear regression model. RMSE is a metric used to 
            evaluate the accuracy of a regression model and represents the average magnitude of 
            the errors in the predictions. SD is a measure of the variability of the data around
            the mean. In the context of evaluating a linear regression model, SD is often used 
            to assess the goodness of fit of the model.
        </p>
        <p>    
            The results of many runs are evaluated to determine the best model. The results of 
            a single pass with { folds } folds are as follows:
        </p>
        <div class="row overflow-auto">
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
        </div>
        <p> Over many of runs it was clear that the best model was the <strong>Gradient boosting</strong> model. </p>

        <hr class="my-5" />
        <h2>Stacking </h2>
        <p> 
            Stacking is a method of combining the results of multiple models. The stacking model is built using the results of the
            best models from the cross fold validation. The results of the stacking model are as follows:
        </p>
        <p><strong>Models used:</strong> {', '.join([x.__class__.__name__ for x in model_list])}</p>
        <p><strong>RMSE this run:</strong> &nbsp; { stacked_rmse :,.2f}</p>
        <p>
            Over { len(stacked_results) } runs the average RMSE was { stacked_mean :,.2f}.
            The Gradient boosting model was the best model so far.
        </p>
        <hr class="my-5" />
        <p class="text-center">Continue the process by further <a class="text-secondary" href="/model-evaluation">evaluating the model</a>.</p>
        <hr class="my-5" />
        <div class="row">
            <div class="{ style.accordion } mb-5" id="code">
                <div class="accordion-item">
                    <h2 class="accordion-header" id="headingOne">
                        <button class="{ style.code_button }" type="button" data-bs-toggle="collapse" data-bs-target="#collapseData">
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

    # Cache the html
    with open('cache/data-modeling.html', 'w') as f:
        f.write(html_str)

    return html_str

 