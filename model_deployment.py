import os, pickle
# Third party imports
from flask import url_for
import pandas as pd
# Local imports
import model
import style

def html(form) -> str:
    ''' Returns the HTML for the page. '''
    result_html = ''
    if form != None:

        # Unpickle the final model
        with open('data/final_model.pkl', 'rb') as f:
            final_model = pickle.load(f)

        input_dict = {
            'KitchenQual_Ex':   1 if form['kitchenQl'] == '1' else 0,   
            'BsmtQual_Ex':      1 if form['bsmnQl'] == '1' else 0,   
            'TotalBathrooms':   int(form['bathrooms']),   
            'YearBuilt':        int(form['year']),   
            'OverallQual':      form['quality'],   
            'TotRmsAbvGrd':     form['rooms'],   
            'OverallCond':      form['condition'],   
            'TotalSquareFeet':  form['totSqFt'],  
            'GarageCars':       form['cars'],   
            'Fireplaces':       form['fireplaces'],   
            'GrLivArea':        form['AbvGrndSqrFt'],   
            'TotalBsmtSF':      form['BsmnSqFt'],
        }

        df = pd.DataFrame(input_dict, index=[0])

        # Get the prediction
        prediction = final_model['Model'].predict(df)

        result_html = f'''
            <hr class="my-5">
            <h3>Results</h3>
            <table class="{style.table}">
                <tr>
                    <th>Year built</th><td>{form['year']}</td>
                    <th>Rooms</th><td>{form['rooms']}</td>
                    <th>Condition</th><td>{form['condition']}</td>
                </tr>
                <tr>
                    <th>Total Sq Ft</th><td>{form['totSqFt']}</td>
                    <th>Bathrooms</th><td>{form['bathrooms']}</td>
                    <th>Quality</th><td>{form['quality']}</td>
                </tr>
                <tr>
                    <th>Bsmn Sq Ft</th><td>{form['BsmnSqFt']}</td>
                    <th>Garage cars</th><td>{form['cars']}</td>
                    <th>Kitchen Quality</th><td>{form['kitchenQl']}</td>
                </tr>
                <tr>
                    <th>Abv Grnd Sq Ft</th><td>{form['AbvGrndSqrFt']}</td>
                    <th>Fireplaces</th><td>{form['fireplaces']}</td>
                    <th>Bsmn Quality</th><td>{form['bsmnQl']}</td>
                </tr>
            </table>
            <p> The predicted selling price of the home is <strong>${ prediction[0] :,.2f}</strong> </p>
        '''
   
    # Get the code for the page
    with open('model_deployment.py', 'r') as f:
        code = f.read().replace('<', '&lt;').replace('>', '&gt;')

    html_str = f'''
        <div class="row mt-5" style="height:300px;">
            <img src="{url_for('static', filename='banner-home.jpg')}" alt="Homes" style="width:100%;height:300px;object-fit: cover;">
        </div>    
        <h1 class="mt-5">Model deployment</h1>    
        <p> 
            Given the results of the model evaluation, we can now deploy the model to a production environment. How do we want to deploy the model?
            Perhaps we want to deploy the model as a web application, or perhaps we want to deploy the model as a REST API. The model can be deployed
            in many different ways. 
        </p>
        <hr class="my-5" />
        <p> This deployment will be a simple example of a web form  where user can submit their home information to and get a predicted selling price. </p>
        <form class="mt-5" action="/model-deployment" method="post">
            <div class="col-2">
                <label for="yearBuilt" class="form-label">Year built</label>
                <input type="number" name="year" class="form-control" id="yearBuilt">
            </div>
            <div class="row mt-3">
                <div class="col-3">
                    <label for="roomAbvGround" class="form-label">Rooms abv grd</label>
                    <input type="number" name="rooms" class="form-control" id="roomAbvGround">
                </div>
                <div class="col-3">
                    <label for="bathrooms" class="form-label">Bathrooms</label>
                    <input type="number" name="bathrooms" class="form-control" id="bathrooms" placeholder="">
                </div>
                <div class="col-3">
                    <label for="garage" class="form-label">Car garage</label>
                    <input type="number" name="cars" class="form-control" id="garage" placeholder="">
                </div>
                <div class="col-3">
                    <label for="fireplaces" class="form-label">Fireplaces</label>
                    <input type="number" name="fireplaces" class="form-control" id="fireplaces">
                </div>
            </div>
            <div class="row mt-3">
                <div class="col-4">
                    <label for="TotalSqFt" class="form-label">Total sq feet</label>
                    <input type="number" name="totSqFt" class="form-control" id="TotalSqFt">
                </div>

                <div class="col-4">
                    <label for="BsmntSqFt" class="form-label">Basement sq feet</label>
                    <input type="number" name="BsmnSqFt"  class="form-control" id="BsmntSqFt" placeholder="">
                </div>
                <div class="col-4">
                    <label for="AbvGrdSqFt" class="form-label">Abv grd sq feet</label>
                    <input type="number" name="AbvGrndSqrFt" class="form-control" id="AbvGrdSqFt" placeholder="">
                </div>
            </div>    
            <div class="row mt-3">

                <div class="col-3">
                    <label for="condition" class="form-label">Overall condition</label>
                    <select name="condition" class="form-select" id="condition">
                        <option selected>Select 1 (low) to 10 (high)</option>
                        <option value="0">One</option>
                        <option value="1">Two</option>
                        <option value="2">Three</option>
                        <option value="3">Four</option>
                        <option value="4">Five</option>
                        <option value="5">Six</option>
                        <option value="6">Seven</option>
                        <option value="7">Eight</option>
                        <option value="8">Nine</option>
                        <option value="9">Ten</option>
                    </select>
                </div>

                <div class="col-3">
                    <label for="quality" class="form-label">Overall quality</label>
                    <select name="quality" class="form-select" id="quality">
                        <option selected>Select 1 (low) to 10 (high)</option>
                        <option value="0">One</option>
                        <option value="1">Two</option>
                        <option value="2">Three</option>
                        <option value="3">Four</option>
                        <option value="4">Five</option>
                        <option value="5">Six</option>
                        <option value="6">Seven</option>
                        <option value="7">Eight</option>
                        <option value="8">Nine</option>
                        <option value="9">Ten</option>
                    </select>
                </div>
                <div class="col-3">
                    <label for="kitchQuality" class="form-label">Kitchen quality</label>
                    <select name="kitchenQl" class="form-select" id="kitchQuality">
                        <option selected>Select one</option>
                        <option value="1">Excellent</option>
                        <option value="2">Good</option>
                        <option value="3">Average</option>
                        <option value="4">Fair</option>
                        <option value="5">Poor</option>
                    </select>
                </div>

                <div class="col-3">
                    <label for="bsmnQuality" class="form-label">Basement quality</label>
                    <select name="bsmnQl" class="form-select" id="bsmnQuality">
                        <option selected>Select one</option>
                        <option value="1">Excellent</option>
                        <option value="2">Good</option>
                        <option value="3">Average</option>
                        <option value="4">Fair</option>
                        <option value="5">Poor</option>
                    </select>
                </div>
            </div>  
            <div class="col-12 mt-5">
                <input type="submit" value="Submit model" class="btn btn-{style.bs_color}"></input>
            </div>
        </form>
        {result_html}
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

    return html_str