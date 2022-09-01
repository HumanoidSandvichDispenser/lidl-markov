import random
import math
from typing import List, Tuple

import spacy
from spacy.tokens.doc import Doc
from spacy.tokens.token import Token
from markovchain import MarkovChain


nlp = spacy.load("en_core_web_sm")


class MoveableChain():
    chain: MarkovChain
    original_prompt: str = ""
    original_prompt_tokens: Doc
    current_state: Tuple[str, ...]

    def set_prompt(self, prompt: str = ""):
        self.original_prompt = prompt
        self.original_prompt_tokens = nlp(prompt)
        self.current_state = MarkovChain.get_pos_tags_from_span(self.original_prompt_tokens)

    def complete(self, min_tokens = 10, max_tokens = 250):
        result = ""
        for i in range(max_tokens):
            next_token = self.move()
            if next_token:
                result += next_token.text_with_ws
                if i >= min_tokens and next_token.text == ".":
                    return result
        return result

    def move(self):
        next_token = self.peek()
        self.current_state = self.current_state + (MarkovChain.get_pos_tag(next_token),)
        return next_token

    def peek(self) -> Token:
        """
        Peeks and chooses a random word to complete the prompt.
        """
        node_contents: List[Token] = []
        node_weights = []

        # grab keywords from the prompt
        prompt_keywords = []
        for token in self.original_prompt_tokens:
            keyword = token.lemma_
            if keyword not in prompt_keywords:
                prompt_keywords.append(keyword)

        # current state is the last FOUR tokens, so apostrophes aren't
        # split badly
        #self.current_state = prompt_doc

        # splits into different weights

        # for a state size of 3, it would be range(1, 4) for i to go from 1, 2,
        # 3, so we would use range(1, state_size + 1).
        #
        # HOWEVER, we also want to loop an extra time. we only use the extra loop
        # if we DON'T find any completions in the chain from the original states.
        tokens = self.current_state
        for i in range(1, self.chain.state_size + 2):
            #state_tokens = tokens[-self.chain.state_size + i - 1:]
            # this grabs the last n words, where n is the state size
            #state = MarkovChain.get_pos_tags_from_span(state_tokens)
            state = self.current_state[-self.chain.state_size + i - 1:]

            # we can't find a completion
            if state not in self.chain.transitions:
                #print("Not in generation")
                # if we're not in that special case, then we continue looping
                if i < self.chain.state_size + 1:
                    #print("CONTINUE!!!")
                    continue
                # if we didn't find a completion, we are in that special case
                elif len(node_contents) == 0:
                    # the CURRENT STATE MUST BE EMPTY to GUARANTEE generation
                    state = ()
                else:
                    #print("What the fuck happened " + str(i))
                    continue
            # if we in that extra loop but don't need to guarantee generation,
            # just skip it
            elif i >= self.chain.state_size + 1:
                #print("Pag manning irl..??")
                continue

            possible_nodes = self.chain.transitions[state]

            for node in possible_nodes:
                node_contents.append(node.content)
                # look for common keywords
                num_common_keywords: int = 0
                for prompt_keyword in prompt_keywords:
                    if prompt_keyword in node.keywords:
                        num_common_keywords += node.keywords[prompt_keyword]

                # lower state size = less weight, so divide by i^2
                # unless it is related
                weight = (1 + pow(num_common_keywords, 3)) / pow(i, 2)

                node_weights.append(weight)

        print(node_contents)
        return random.choices(node_contents, node_weights, k = 1)[0]
