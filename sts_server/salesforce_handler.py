# salesforce_handler.py

import os
from simple_salesforce import Salesforce

class SalesforceHandler:
    def __init__(self):
        self.sf = None
        try:
            print("[INFO] Connecting to Salesforce...")
            print(os.getenv('SF_USERNAME'))
            print(os.getenv('SF_PASSWORD'))
            print(os.getenv('SF_SECURITY_TOKEN'))
            SF_USERNAME = os.getenv('SF_USERNAME')
            SF_PASSWORD = os.getenv('SF_PASSWORD')
            SF_TOKEN = os.getenv('SF_SECURITY_TOKEN')
            self.sf = Salesforce(username=SF_USERNAME, password=SF_PASSWORD, security_token=SF_TOKEN)
            print("[INFO] Successfully connected to Salesforce.")
        except Exception as e:
            print(f"[ERROR] Salesforce connection failed: {e}")
            raise

    def get_session(self):
        return self.sf