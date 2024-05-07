from pydantic import BaseModel


class Issue(BaseModel):
    id: int


class IssuesList(BaseModel):
    issues: list[Issue]
    total_count: int
    offset: int
    limit: int
