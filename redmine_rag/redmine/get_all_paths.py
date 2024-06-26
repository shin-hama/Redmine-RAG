import os
from typing import Iterator

from dotenv import load_dotenv

from redmine_rag.redmine.client import RedmineClient

load_dotenv()  # take environment variables from .env.
api_key = os.environ.get("REDMINE_API_KEY")
if api_key is None:
    raise ValueError("Please set the REDMINE_API_KEY environment variable.")

redmine_url = os.environ.get("REDMINE_URL")
if redmine_url is None:
    raise ValueError("Please set the REDMINE_URL environment variable.")


def get_all_paths() -> Iterator[str]:
    redmine = RedmineClient(
        redmine_url,
        api_key,
    )

    for issue in redmine.all_issues().issues:
        yield f"{redmine_url}/issues/{issue.id}"


if __name__ == "__main__":
    for p in get_all_paths():
        print(p)
