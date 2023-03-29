import os, pickle
# Third party imports
from flask import url_for
import pandas as pd
# Local imports
import model
import style

def html() -> str:
    ''' Returns the HTML for the page. '''

    # # Load a cached version of the page if it exists
    # if os.path.exists('cache/model-evaluation.html'):
    #     with open('cache/model-evaluation.html', 'r') as f:
    #         return f.read()s

    # Create an array of the models and 100 results for each model

    # unpickle the model
    with open('data/model_list.pkl', 'rb') as f:
        model_list = pickle.load(f)  

    # unpickle the data
    with open('data/df_keep.pkl', 'rb') as f:
        X = pickle.load(f)

    with open('data/target.pkl', 'rb') as f:
        y = pickle.load(f)

    # Check if the model summary has been cached
    if os.path.exists('data/model_summary.pkl'):
        with open('data/model_summary.pkl', 'rb') as f:
            result_df = pickle.load(f)
    else:
        # Loop 20 times to get 100 results with 5 folds
        results = {}
        for mod in model_list:
            results[mod.__class__.__name__] = []

            for _ in range(20):
                result = model.evaluate_model(mod, None, X, y, folds=5)
                results[mod.__class__.__name__] += [ x['RMSE'] for x in result]

        # Create a dataframe of the results
        result_df = pd.DataFrame(results)

        # pickle the dataframe
        with open('data/model_summary.pkl', 'wb') as f:
            pickle.dump(result_df, f)
    
    # Build the svgs
    rmse_svg = model.svg_rmse(result_df, moving_average=10)
    std_svg  = model.svg_rmse(result_df, std_dev=True, moving_average=20)

    final_model = {'Model': model_list[-1], 'Avg. RMSE': result_df['GradientBoostingRegressor'].mean(), 'Std. Dev': result_df['GradientBoostingRegressor'].std()}

    # pickle the final model
    with open('data/final_model.pkl', 'wb') as f:
        pickle.dump(final_model, f)

    # Load the code preview for the page
    with open('model_evaluation.py', 'r') as f:
        code = f.read().replace('<', '&lt;').replace('>', '&gt;')
    html_str = f'''
        <div class="row mt-5" style="height:300px;">
            <img src="{url_for('static', filename='banner-home.jpg')}" alt="Homes" style="width:100%;height:300px;object-fit: cover;">
        </div>    
        <h1 class="mt-5">Data model evaluation</h1>
        <p> 
            Even after fairly consclusive results about the model to use, it's a good idea to go yet further evaluations.
        </p>
        <hr class="my-5" />
        <h2>Root mean squared error</h2>
        <p>
            The root mean squared error (RMSE) is a measure of how well the model predicts the target variable.
            The lower the RMSE, the better the model.
        </p>
        <div class="row mt-5">
            {rmse_svg}
        </div>
        <hr class="my-5" />
        <h2>Standard deviation</h2>
        <p>
            The standard deviation is a measure of how spread out the data is from the mean.
            A low standard deviation means that the data is close to the mean.
        </p>
        <div class="row mt-5">
            {std_svg}
        </div>
        <hr class="my-5" />
        <h2>Evaluation conclusion</h2>
        <p>
            After running 100 evaluations of each of the models the best model is clearly the <strong>Gradient Boosting Regressor</strong>.
            Particularly when looking at the RMSE plot, the Gradient Boosting Regressor has the lowest by a significant margin. And while 
            the standard deviation is around average, the difference in RMSE more than makes up for this. 
        </p>
        <p> The final model is: { final_model["Model"]} with an average RMSE of { final_model["Avg. RMSE"] :,.2f} and a standard deviation of { final_model["Std. Dev"] :,.2f}</p>
        <hr class="my-5" />
        <p class="text-center"> Onto a <a class="text-secondary" href="/model-deployment">model deployment</a>. </p>
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
    with open('cache/model-evaluation.html', 'w') as f:
        f.write(html_str)


    return html_str