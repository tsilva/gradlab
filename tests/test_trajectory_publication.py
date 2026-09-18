from types import SimpleNamespace


class Hub:
    """Controlled remote contents, including compare-and-swap and lost replies."""

    def __init__(self):
        self.files = {}
        self.head = "0"
        self.lose_reply = False
        self.conflict_once = False

    def list_repo_tree(self, *args, **kwargs):
        return [SimpleNamespace(path=key) for key in self.files]

    def repo_info(self, **kwargs):
        return SimpleNamespace(sha=self.head)

    def read(self, path, revision):
        assert revision == self.head
        return self.files.get(path)

    def create_commit(self, **kwargs):
        if self.conflict_once:
            self.conflict_once = False
            self.head = str(int(self.head) + 1)
            raise RuntimeError("remote head changed")
        assert kwargs["parent_commit"] == self.head
        for op in kwargs["operations"]:
            from pathlib import Path

            value = op.path_or_fileobj
            self.files[op.path_in_repo] = (
                value if isinstance(value, bytes) else Path(value).read_bytes()
            )
        self.head = str(int(self.head) + 1)
        if self.lose_reply:
            self.lose_reply = False
            raise RuntimeError("commit response lost")
        return SimpleNamespace(oid=self.head)
