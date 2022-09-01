from typing import Dict

from spacy.tokens.token import Token


class MarkovNode():
    content: Token
    keywords: Dict[str, int]

    def __init__(self, content) -> None:
        self.content = content

    def __str__(self) -> str:
        return self.content.text

    def __repr__(self) -> str:
        return self.content.text + "(%d)" % len(self.keywords)
