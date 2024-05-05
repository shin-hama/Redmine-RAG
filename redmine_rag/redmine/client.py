from typing import Iterator

from redminelib import Redmine
from redminelib.resources import Issue


class RedmineClient:
    def __init__(self, url, key):
        self.redmine = Redmine(url, key=key)

    def all_issues(self) -> Iterator[Issue]:
        return self.redmine.issue.all().values()
