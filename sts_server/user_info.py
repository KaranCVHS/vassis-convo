# user_info.py

class UserInfo:
    """A class to hold the state for a single user session."""
    def __init__(self):
        self.first_name = ""
        self.last_name = ""
        self.is_verified = False
        self.sf_data = None
        self.messages = []