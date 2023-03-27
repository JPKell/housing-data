from html import escape

def html() -> str:
    ''' Reads the main.py file and returns the contents as a string '''
    html_str = ''

    with open('model.py') as f:
        main = f.read()
        html_str = f'<pre>{escape(main)}</pre>'
    return html_str
    