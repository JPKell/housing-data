import sys
sys.path.insert(0, '/var/www/jordankell.com/housing_data')

activate_this = '/var/www/jordankell.com/housing_data/venv/bin/activate'
with open(activate_this) as file_:
    exec(file_.read(), dict(__file__=activate_this))

from main import app as application