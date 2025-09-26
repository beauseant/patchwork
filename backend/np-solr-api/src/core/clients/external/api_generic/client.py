import logging
import os

import requests


class Client(object):
    def __init__(self, logger: logging.Logger, logger_name: str) -> None:
        """
        Parameters
        ----------
        logger : logging.Logger
            The logger object to log messages and errors.
        """

        if logger:
            self.logger = logger
        else:
            import logging
            logging.basicConfig(level='DEBUG')
            self.logger = logging.getLogger('Inferencer')
        return
    
    def _do_request(self,
                    type: str,
                    url: str,
                    timeout: int = 10,
                    **params) -> requests.Response:
        """Sends a request to an API.

        Parameters
        ----------
        type : str
            The type of the request.
        url : str
            The URL of the Inferencer API.
        timeout : int, optional
            The timeout of the request in seconds, by default 10.
        **params: dict
            The parameters of the request.
            
        Returns
        -------
        requests.Response: requests.Response
            An object of the requests.Response class.
        """

        # Send request
        if type == "get":
            resp = requests.get(
                url=url,
                timeout=timeout,
                **params
            )
            pass
        elif type == "post":
            resp = requests.post(
                url=url,
                timeout=timeout,
                **params
            )
        else:
            self.logger.error(f"-- -- Invalid type {type}")
            return

        return resp