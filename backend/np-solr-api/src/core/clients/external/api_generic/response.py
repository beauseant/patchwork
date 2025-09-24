import logging
import requests


class Response(object):
    """
    A class to handle genereic API response and errors.
    """

    def __init__(self,
                 resp: requests.Response,
                 logger: logging.Logger) -> None:

        # Get JSON object of the result        
        resp = resp.json()

        self.status_code = resp['responseHeader']['status']
        self.time = resp['responseHeader']['time']
        self.results = resp['response']

        if self.status_code == 200:
            logger.info(f"-- -- Inferencer request acknowledged")
        else:
            logger.info(
                f"-- -- Inference request generated an error: {self.results['error']}")
        return