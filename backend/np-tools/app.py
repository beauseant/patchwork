"""Main application entry point
"""
import logging
from src.apis import api
from flask import Flask
from pyfiglet import figlet_format
from termcolor import cprint

numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)

# Create Flask app
app = Flask(__name__)
# Deactivate the default mask parameter.
app.config["RESTX_MASK_SWAGGER"] = False
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 
api.init_app(app)

if __name__ == '__main__':
    cprint(figlet_format("NP Tools API",
           font='big'), 'blue', attrs=['bold'])
    print('\n')    
    
    app.run(host='0.0.0.0', port=102, debug=True)
    #from waitress import serve
    #serve(app, host="0.0.0.0", port=82)
