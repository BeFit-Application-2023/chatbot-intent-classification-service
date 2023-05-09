# Importing all needed libraries.
import json
import hmac
import hashlib


class SecurityManager:
    def __init__(self, key : str) -> None:
        '''
            This function creates and sets up the Security Manager.
                :param key: str
                    The secret key of the service used for HMAC.
        '''
        self.key = str.encode(key)

    def __encode_hmac(self, request_body) -> str:
        '''
            This function calculates the HMAC of the request body and returns it.
                :param request_body: dict
                    The body of the request.
                :return json_hmac: str
                    The HMAC of the request body.
        '''
        request_body_binary = json.dumps(request_body).encode()
        json_hmac = hmac.new(self.key, request_body_binary, hashlib.sha256).hexdigest()

        return json_hmac

    def verify(self, token : str, request_body : dict) -> bool:
        '''
            This function authenticates the request body.
                :param token: str
                    The token sent with the request from the headers.
                :param request_body: dict
                    The body of the request.
        '''
        # Compuiting the HMAC o the request body.
        request_hmac = self.__encode_hmac(request_body)

        # Verifying the request HMAC.
        if token == request_hmac:
            return True
        else:
            return False

    def check_access_token(self, header_dict : dict):
        '''
            This function check is the authentication token is present.
                :param header_dict: dict
                    The header dictionary of the request.
        '''
        if "Token" not in header_dict:
            return {
                "message" : "Missing Authorization token!",
                "code" : 401
            }
        else:
            return "OK"

    def check_request(self, request):
        '''
            This function implements the HMAC authentication of the request.
        '''
        # Checking the presence of the authentication token.
        check_response = self.check_access_token(dict(request.headers))

        if check_response != "OK":
            return check_response
        elif not self.verify(request.headers["token"], request.json):
            # If the request didn't passed the HMAC authentication a 401 status error code is returned.
            return {
                "message" : "401 Unauthorized",
                "code" : 401
            }
        else:
            return "OK"

