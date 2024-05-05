import os
from typing import Iterator

from dotenv import load_dotenv

from .client import RedmineClient

load_dotenv()  # take environment variables from .env.
api_key = os.environ.get("REDMINE_API_KEY")
if api_key is None:
    raise ValueError("Please set the REDMINE_API_KEY environment variable.")


def get_all_paths() -> Iterator[str]:
    redmine = RedmineClient(url="https://redmine.langchain.com", key=api_key)
    for issue in redmine.all_issues():
        if issue.url is not None:
            yield issue.url
