from html import escape

def html() -> str:
    ''' Reads the main.py file and returns the contents as a string '''
    html_str = ''

    with open('model.py') as f:
        main = f.read()
        html_str = f'''
            <h1>Additional code</h1>
            <p>The previous pages pulled heavily from this code. Any statistics imports were
               done here to keep the statistics calls all to model from the pages. </p>
            <hr class="my-5" />
            <pre>{escape(main)}</pre>'''
    return html_str
    