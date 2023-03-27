# Std lib imports
import os
# Third party imports
from flask import url_for
# Local imports
import model
import style


def html() -> str:
    ''' Returns the HTML for the page. '''

    # Check for a cached version of the data and return it if it exists
    if os.path.exists('cache/data-exploration.html'):
        with open('cache/data-exploration.html', 'r') as f:
            return f.read()

    # Basic data loading and summaries
    df = model.get_df_from_csv()
    
    # Id had no predictive power, so we drop it
    df.drop('Id', axis=1, inplace=True)

    keys = model.get_keys(df)
    nulls = model.get_nulls(df)
    non_numeric = model.get_non_numeric_columns(df)
    
    # Get features and target
    X, y = model.get_features_target(df)

    # Generate sales histogram SVG
    sales_hist = model.svg_histogram(y, bins=30, x_lab='Sale Price', y_lab='Count', title='Sales Price Histogram')

    # Pass numerical columns to csv
    numerical = df.select_dtypes(include=['int64', 'float64'])
    model.write_df_to_csv(numerical, 'data/correlation.csv')
    model.process_r_code('correlation.R')

    # Get correlation results
    corr_df = model.get_correlation_results()

    # Get the top correlated features
    target_corr = corr_df[['Feature', 'SalePrice']].sort_values(by='SalePrice',ascending=False)
    target_corr = target_corr[1:]
    target_corr.dropna(inplace=True)

    # # Get scatter plots
    pos_corr_plots = []
    active = 'active'
    for i in range(0, 6):
        feature = target_corr.iloc[i]['Feature']
        # R Adds an X infront of numerical columns, so we remove it
        feature = feature[1:] if feature[0] == 'X' else feature
        svg = model.svg_scatter(X[feature], y, title=feature, y_lab='Sale Price')
        pos_corr_plots.append(svg)

    neg_corr_plots = []
    active = 'active'
    for i in range(1, 7):
        feature = target_corr.iloc[-i]['Feature']
        # R Adds an X infront of numerical columns, so we remove it
        feature = feature[1:] if feature[0] == 'X' else feature
        svg = model.svg_scatter(X[feature], y, title=feature, y_lab='Sale Price')

        html_str = f'''                            
            <div class="carousel-item {active} justify-content-center">
                <div class="d-flex justify-content-center">
                    { svg }
                </div>
            </div>'''
        active = ''
        neg_corr_plots.append(html_str)

    html_str = f'''
            <div class="row mt-5" style="height:300px;">
                <img src="{url_for('static', filename='banner-home.jpg')}" alt="Homes" style="width:100%;height:300px;object-fit: cover;">
            </div>
            <div class="row mt-5">
                <div class="col-12">
                    <h1>Data exploration</h1>
                    <p> The data originates from a Kaggle practice set 
                    (<a href="https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data">available here</a>) 
                    and contains {len(keys)} columns and {len(df)} rows. The data gathered covers a fair range of attributes you would 
                    expect to see on a real estate listing. Many of these values could be scraped from online listings for an automated
                    system to detect prime listings. I will leave that implementation to the end user should the want to.</p>

                        <p> {len(keys)} columns is a lot to look at, let alone gather enough data for. Therefore I will reduce this 
                    to a manageable number. Something that a human might be expected to manually input. For details about 
                    the columns see the drop down below.  </p>


                    <p> An initial look at the data shows a few areas which will need to be addressed before we can build a model. </p>
                    <ul>
                        <li> Id - This is a unique identifier for each row and is not valuable </li>
                        <li> MSSubClass - This is a categorical value and not numeric </li>
                        <li> Columns with similare meaning, e.g. overall quality and condition, full bath, half bath </li>
                        <li> There are { len(nulls) } numeric columns with missing values. See drop down below for the list</li>
                        <li> There are { len(non_numeric)} columns which are not numeric. These will have to be one-hot encoded </li>
                    </ul>
                    <p>For more information about the columns and their meaning please see <a href="/data-description">the Kaggle descriptions</a></p>
                </div>
                <hr class="my-5" />
                <h2>Exploring the data</h2>
                <p> Below are some summary tables representing the dataset and some statistical information about the columns. </p>
                <div class="{ style.accordion } mb-5" id="dfDetails">
                    <div class="accordion-item">
                        <h2 class="accordion-header" id="headingOne">
                            <button class="{ style.accordion_button }" type="button" data-bs-toggle="collapse" data-bs-target="#collapseData">
                                Data preview
                            </button>
                        </h2>
                        <div id="collapseData" class="accordion-collapse collapse show" data-bs-parent="#dfDetails">
                            <div class="accordion-body">
                                <div class="row overflow-auto">
                                    {df.head(5).to_html(classes=style.table)}
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="accordion-item">
                        <h2 class="accordion-header" id="headingOne">
                            <button class="{ style.accordion_button }" type="button" data-bs-toggle="collapse" data-bs-target="#collapseSummary">
                                Column statistical summary
                            </button>
                        </h2>
                        <div id="collapseSummary" class="accordion-collapse collapse" data-bs-parent="#dfDetails">
                            <div class="accordion-body">
                                <div class="row overflow-auto">
                                    {df.describe().T.to_html(classes=style.table)}
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="accordion-item">
                        <h2 class="accordion-header" id="headingOne">
                            <button class="{ style.accordion_button }" type="button" data-bs-toggle="collapse" data-bs-target="#collapseObject">
                                Non-numerical columns
                            </button>
                        </h2>
                        <div id="collapseObject" class="accordion-collapse collapse" data-bs-parent="#dfDetails">
                            <div class="accordion-body">
                                <div class="row overflow-auto">
                                    {',&nbsp; '.join(non_numeric)}
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="accordion-item">
                        <h2 class="accordion-header" id="headingOne">
                            <button class="{ style.accordion_button }" type="button" data-bs-toggle="collapse" data-bs-target="#collapseNulls">
                                Column null counts
                            </button>
                        </h2>
                        <div id="collapseNulls" class="accordion-collapse collapse" data-bs-parent="#dfDetails">
                            <div class="accordion-body">
                                <div class="row overflow-auto">
                                    {nulls.to_frame().to_html(classes=style.table, header=False)}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                <hr class="my-5" />
                <div class="row">
                    <div class="col-lg-4 col-12 mt-3 bg-{style.bs_color}">
                        <h2 class="mt-2 text-white">Sales price</h2>
                        <p class="mt-4 text-white"> We want to know the selling price of a home. The data that we have skews to the right, so we will need to
                            transform the data to make it more normaly distributed. </p>
                    </div>
                    <div class="col-lg-8 col-12">
                        {sales_hist}
                    </div>
                </div>
                <hr class="my-5" />
                <div class="row overflow-auto">
                    <h2>Feature correlation</h2>
                    <p> Using R we can see the correlation between the features. The correlation is a value between -1 and 1. A value of 1
                        means that the features are perfectly correlated. A value of -1 means that the features are perfectly negatively correlated.
                        A value of 0 means that there is no correlation between the features. </p>

                    <h3 class="mt-3">Positive correlations</h3>
                    {target_corr.head(6).T.to_html(header=False, classes=style.table)}
                </div>
                <div class="row mt-3">
                    <div id="posCorrCarousel" class="carousel carousel-dark carousel-fade slide" data-bs-ride="carousel">
                        <div class="carousel-inner">
                            <div class="carousel-item active justify-content-center">
                                <div class="d-flex justify-content-center">
                                    { pos_corr_plots[0] }
                                </div>
                                <p> As <strong>overall quality</strong> increases the margin of error increases more significantly. 
                                    This would be a great variable to bin with a binary outcome.</p>
                            </div>

                            <div class="carousel-item mx-auto">
                                <div class="d-flex justify-content-center">
                                    { pos_corr_plots[1] }
                                </div>
                                <p><strong> Above ground living area</strong> is highly correlated with the <strong>sale price</strong>. 
                                    This is a great feature to use in our model.</p>
                            </div>

                            <div class="carousel-item">
                                <div class="d-flex justify-content-center">
                                    { pos_corr_plots[2] }
                                </div>
                                <p><strong>Number of car garage</strong> shows a fairly large standard deviation, Might not be as valuable as the correlation infers</p>
                            </div>

                            <div class="carousel-item">
                                <div class="d-flex justify-content-center">
                                    { pos_corr_plots[3] }
                                </div>
                                <p><strong>Garage area</strong> hoever has a strong correlation to the <strong>sale price</strong> and does not vary as widely. 
                                    Based on this and the fact that the previous feature is highly correlated to this one, we can drop the previous feature. 
                                    Some feature engineering will be required to make this feature more useful as houses without garages will have a value of 0.
                                </p>
                            </div>

                            <div class="carousel-item">
                                <div class="d-flex justify-content-center">
                                    { pos_corr_plots[4] }
                                </div>
                                <p><strong>Basement sq. feet</strong> out performs first floor sq. feet. Again some feature engineering will be required to 
                                make this feature more useful as houses without basements will have a value of 0. Therefore it may be worth using 1st floor
                                sq. feet instead of this one.</p>
                            </div>

                            <div class="carousel-item">
                                <div class="d-flex justify-content-center">
                                    { pos_corr_plots[5] }
                                </div>
                                <p><strong>1st floor sq. feet</strong> has a very similar correlation as basement sq. feet. It's also an easier value it 
                                input into a model rather than having to impute one.</p>
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
                <div class="row mt-5 overflow-auto">
                    <h3>Negative correlations</h3>
                    <p> A strong negative correlation is a valuable feature also. However it doesn't look like there are any strong negative 
                        correlations of the lowest of the correlation scores</p>
                    {target_corr.tail(6).T.to_html(header=False, classes=style.table)}
                </div>
                <div class="row my-3">
                    <div id="carouselExampleCaptions" class="carousel carousel-dark carousel-fade slide" data-bs-ride="carousel">
                        <div class="carousel-inner">
                            { ''.join(neg_corr_plots) }
                        </div>
                        <button class="carousel-control-prev" type="button" data-bs-target="#carouselExampleCaptions" data-bs-slide="prev">
                            <span class="carousel-control-prev-icon" aria-hidden="true"></span>
                            <span class="visually-hidden">Previous</span>
                        </button>
                        <button class="carousel-control-next" type="button" data-bs-target="#carouselExampleCaptions" data-bs-slide="next">
                            <span class="carousel-control-next-icon" aria-hidden="true"></span>
                            <span class="visually-hidden">Next</span>
                        </button>
                    </div>
                    <p>Overall these are all poor negativly correlated features</p>
                </div>
                <hr />
                <div class="row mt-3">
                    <p>Now armed with this information we continue to <a href="/data-preparation">prepare the data</a></p>
                </div>
            </div>
''' 

    with open('cache/data-exploration.html', 'w') as f:
        f.write(html_str)

    return html_str

    