from flask import url_for
def html() -> str:
    ''' Returns the HTML for the page. '''

    
    html_str = f'''
    <div class="row mt-5" style="height:300px;">
        <img src="{url_for('static', filename='banner-home.jpg')}" alt="Homes" style="width:100%;height:300px;object-fit: cover;">
    </div>
    <h1 class="my-5">Housing market analysis</h1>
        <p> 
            When looking for a dependable investment, there are few more reliable than real estate. The return on your 
            investment depends greatly on the initial value of the property relative to it's "true worth" or what most
            people are willing to pay. Using thousands of records from Iowa's Ames Assessor's Office I built a model to 
            predict the selling price. With a model for expected selling price in hand we can input market data and flag 
            homes listed for below what the model expects the home to sell for. If only it were 2010 and I lived in Iowa. 
        </p>
    <div class="row mt-5">
        <div class="col">   
            <div class="row d-flex justify-content-center my-3">
                <img src="{url_for('static', filename='salesPrice.svg')}" alt="Sales price" style="width:80%">
            </div>
            <p>
                One house or another, what is the difference? Without hard earned experience, it's difficuly to determine 
                which house is a deal and which isn't. Thankfully with data we can let the market be the judge. The market
                in Iowa at the time was significantly below real eastate prices today with the mean home worth around
                $150,000 USD.   
            </p>
        </div>
        <div class="col-3 d-none d-md-block ps-2">
            <img class="mt-4" src="{url_for('static', filename='pink-blue-homes.jpg')}" alt="Homes" style="width:100%">
        </div>
    </div>
    <hr class="my-5" />
    <div class="row">
        <div class="col-12 col-xl-5 text-center">
            <p>
                The dataset comes 80 columns, and while the housing market is complex, there have to be some key features that
                can help the model predict more accurately. Looking at the correlation between features and the sales price 
                helps to select the features to built the model. 
            </p>
            <p>
                The final model uses 12 features, many make sense intuitively like overall condition and quality. Other 
                indicators include how many fireplaces, bathrooms and car garages. These all make sense as a home with more 
                fireplaces or bathrooms indicates a higher end home. This is clear when you look at the scatter plot of 
                the features against the regression line. 
            </p>
        </div>
        <div class="col-12 col-xl-7">
            <div id="posCorrCarousel" class="carousel carousel-dark carousel-fade slide" data-bs-ride="carousel">
                <div class="carousel-inner">
                    <div class="carousel-item active justify-content-center">
                        <div class="d-flex justify-content-center">
                            <img src="{url_for('static', filename='1stFlrSF.svg')}" alt="Sales price" style="width:80%">
                        </div>
                    </div>
                    <div class="carousel-item mx-auto">
                        <div class="d-flex justify-content-center">
                            <img src="{url_for('static', filename='GrLivArea.svg')}" alt="Sales price" style="width:80%">
                        </div>
                    </div>
                    <div class="carousel-item">
                        <div class="d-flex justify-content-center">
                            <img src="{url_for('static', filename='GarageArea.svg')}" alt="Sales price" style="width:80%">
                        </div>
                    </div>
                    <div class="carousel-item">
                        <div class="d-flex justify-content-center">
                            <img src="{url_for('static', filename='OverallQual.svg')}" alt="Sales price" style="width:80%">
                        </div>
                    </div>
                </div>
                <button class="carousel-control-prev" type="button" data-bs-target="#posCorrCarousel" data-bs-slide="prev">
                    <span class="carousel-control-prev-icon" aria-hidden="true"></span>
                    <span class="visually-hidden">Previous</span>
                </button>
                <button class="carousel-control-next" type="button" data-bs-target="#posCorrCarousel" data-bs-slide="next">
                    <span class="carousel-control-next-icon" aria-hidden="true"></span>
                    <span class="visually-hidden">Next</span>
                </button>
            </div>
        </div>
    </div>
    <hr class="my-5" />
    <div class="row">
        <p>
            Not every feature was so clear as to how it would help the model. Converting categorical features into a
            numerical column the models could use yielded 155 total columns which were then sorted through automatically
            taking the results of three different feature selection algorithms.  
        </p>
        <p>
            In the end a variety of features, binned, engineered, and raw got fed into six different models. Which then were
            fed into a stacking algorithm. And in the end the Gradient Boosting Regressor proved the most effective for this
            data ste. Using the root mean squared error statistic to judge the models showed clearly which would perform
            best for the task. 
        </p>
        <div class="d-flex justify-content-center my-5">
            <img src="{url_for('static', filename='rmse.svg')}" alt="Sales price" style="width:80%">
        </div>
    </div>
    <hr class="my-5" />
    <p> 
        A note about the site. In the real world this dataset would be updated often with fresh data which would impact the model.
        Each page has the code behind it to do every transformation and calculation discussed. The code is available at the bottom
        of each page. Since many of these workflows take time the data is run once and the output is cached. You can reset the cache,
        but be aware this will make the first load of each page slower.  
    </p>      
    <div class="d-flex justify-content-center">
        <button class="btn btn-warning" onclick="clearCache()">Clear cache</button>
    </div>
    <hr class="my-5" />
    <div class="row d-flex justify-content-even text-center">
        <div class="col"><a class="text-secondary" href="/data-exploration"> Learn more about the model</a></div>
        <div class="col"><a class="text-secondary" href="/model-deployment"> Jump to the deployment</a></div>
    </div>

    '''


    return html_str


'''


'''