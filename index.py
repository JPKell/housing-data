from flask import url_for
def html() -> str:
    ''' Returns the HTML for the page. '''

    
    html_str = f'''
    <div class="row mt-5" style="height:300px;">
        <img src="{url_for('static', filename='banner-home.jpg')}" alt="Homes" style="width:100%;height:300px;object-fit: cover;">
    </div>
    <div class="row mt-5">
        <div class="col">   
            <h1>Housing market analysis</h1>
            <p> When looking for a dependable investment, there are few more reliable than real estate. The return on your 
                investment depends greatly on the initial value of the property relative to it's "true worth" or what most
                people are willing to pay. By using thousands of existing listings and their final selling price we build 
                a model which will help to determine the expected selling price for a home. With an expected selling price 
                in hand we can input real market data and flag homes listing for below what the model expects the home to 
                sell for. </p>
            <p> This is either a brilliant idea to make lots of money or a direct route to finding the lemons of the real 
                estate market. Either way, spend your own money with whatever level of disernment you are comfortable with. </p>
                
            <p> The page is broken down into 6 sections. </p>
            <ul>
                <li> <a href="/data-exploration">Data exploration</a> 
                    <ul><li>This section looks at the data and how it is structured.</li></ul> 
                </li>
                <li> <a href="/data-preparation">Data preparation</a> 
                    <ul><li>Here we look at the ways the data was transformed for inproved model accuracy </li></ul> 
                </li>
                <li> <a href="/data-modeling">Data modeling</a> 
                    <ul><li> The model is built in this section and we explore different options for building out the model </li></ul> 
                </li>
                <li> <a href="/model-evaluation">Model evaluation</a> 
                    <ul><li>Using the models we created previously we examine how effective they are at price prediction </li></ul> 
                </li>
                <li> <a href="/model-deployment">Model deployment</a> 
                    <ul><li>We deploy the model as a web application and use it to predict the price of a home </li></ul> 
                </li>           
                <li> <a href="/code">Code</a> 
                    <ul><li>The code for this project is available here. </li></ul> 
                </li>
            </ul>
        </div>
        <div class="col-3 d-none d-md-block mt-5 ps-2">
            <img src="{url_for('static', filename='pink-blue-homes.jpg')}" alt="Homes" style="width:100%">
        </div>
    </div>
    '''

    return html_str