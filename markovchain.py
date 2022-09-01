import spacy
import math
from typing import Dict, List, Tuple

from spacy.tokens.doc import Doc
from spacy.tokens.span import Span
from spacy.tokens.token import Token

from markovnode import MarkovNode


nlp = spacy.load("en_core_web_sm")


class MarkovChain():
    transitions: Dict[Tuple[str], List[MarkovNode]] = { }
    state_size: int

    keyword_pos_tags = [ "PROPN", "ADJ", "NOUN" ]

    def __init__(self, state_size: int = 2) -> None:
        self.state_size = state_size
        pass

    @staticmethod
    def get_pos_tag(token: Token) -> str:
        return token.pos_ + "::" + token.text_with_ws

    @staticmethod
    def get_pos_tags_from_span(tokens: Span | Doc) -> Tuple[str, ...]:
        prev_states = []
        for token in tokens:
            # here we store the part of speech with the state, so
            # sentences are more coherent.
            prev_states.append(MarkovChain.get_pos_tag(token))
        return tuple(prev_states)

    def add_chain(self, tokens: Doc):
        # we have to loop twice: first to get ALL the keywords, and second to add
        # a transition in the chain
        keywords: Dict[str, int] = { }
        for i, token in enumerate(tokens):
            if token.pos_ in self.keyword_pos_tags:
                keyword = token.lemma_
                if keyword not in keywords:
                    # count the number of times this token appeared
                    frequency = len(list(filter(lambda t: t.text == token.text,
                                                list(tokens))))
                    keywords[keyword] = frequency
        for i, token in enumerate(tokens):
            for n in range(1, self.state_size + 1):
                prev_tokens = tokens[i - n:i]
                prev_states = self.get_pos_tags_from_span(prev_tokens)
                # if we are at not the beginning of the sentence, and the
                # prev_state is a "beginning of sentence" state, then do not add
                # transition from beginning of sentence to this word/token
                if len(prev_states) == 0 and i > 0:
                    continue
                self.add_transition(tuple(prev_states), token, keywords)

    def add_transition(self,
                       prev_state: Tuple[str],
                       next_state: Token,
                       keywords: Dict[str, int]):
        node = MarkovNode(next_state)
        node.keywords = keywords
        if prev_state in self.transitions:
            self.transitions[prev_state].append(node)
        else:
            self.transitions[prev_state] = [ node ]
