import requests


class RedmineClient:
    def __init__(self, host, key):
        self.host = host
        self.key = key

    def _get(self, url):
        return requests.get(
            url,
            headers={"X-Redmine-API-key": self.key},
            proxies={
                "http": "",
                "https": "",
            },
        )

    def all_issues(self):
        return requests.get(
            f"{self.host}/users/current.json",
        ).text

    def current_user(self):
        """
        クライアントに移譲されたユーザー情報を取得します。
        主にテスト用
        """
        return self._get(
            f"{self.host}/users/current.json",
        ).text
