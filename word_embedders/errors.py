# This file defines the errors of the package.

class NotAValidVersion(Exception):
    '''
        This error is raised when the user tries to create a word embedder
        with an invalid version of this method.
    '''
    pass

class ImpossibleDimension(Exception):
    '''
        This error is raised when the user tries to create a word embedder
        with an invalid vector dimension for this method.
    '''
    pass