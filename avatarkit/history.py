"""Undo / redo for the customizer: snapshots of the option indices."""


class History:
    def __init__(self, state, limit=200):
        self.limit = limit
        self.items = [dict(state)]
        self.index = 0

    @property
    def can_undo(self):
        return self.index > 0

    @property
    def can_redo(self):
        return self.index < len(self.items) - 1

    def commit(self, state):
        """Record `state` if it differs from the current snapshot; returns True if it did."""
        if state == self.items[self.index]:
            return False
        del self.items[self.index + 1:]           # a new edit discards the redo branch
        self.items.append(dict(state))
        if len(self.items) > self.limit:
            del self.items[0]
        self.index = len(self.items) - 1
        return True

    def undo(self):
        if not self.can_undo:
            return None
        self.index -= 1
        return dict(self.items[self.index])

    def redo(self):
        if not self.can_redo:
            return None
        self.index += 1
        return dict(self.items[self.index])
