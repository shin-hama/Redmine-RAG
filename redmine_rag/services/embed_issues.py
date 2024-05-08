from redmine_rag.embedding.embedding import vectorize_webpages
from redmine_rag.redmine.get_all_paths import get_all_paths


def embed_issues():
    vectorize_webpages(list(get_all_paths()))


if __name__ == "__main__":
    embed_issues()
