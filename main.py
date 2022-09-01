from markovchain import MarkovChain
from moveablechain import MoveableChain
import spacy
import time


nlp = spacy.load("en_core_web_sm")

def main():
    corpus: str = ""
    with open("corpus.txt", "r") as f:
        corpus = f.read()
    raw_paragraphs = corpus.split("\n\n")
    paragraphs = []
    for paragraph in raw_paragraphs:
        paragraph = paragraph.strip()
        paragraph = paragraph.replace("\n", " ")
        paragraphs.append(paragraph)
    chain = MarkovChain(3)
    start = time.time()
    for paragraph in paragraphs:
        doc = nlp(paragraph)
        chain.add_chain(doc)
        #print(chain.transitions)
    #print(chain.transitions[('NOUN::amount ', 'ADP::of ')])
    end = time.time()
    print("Time to generate chain: %f seconds" % (end - start))
    moveable_chain = MoveableChain()
    moveable_chain.chain = chain
    moveable_chain.set_prompt("How do you learn a language effectively?")
    print(moveable_chain.complete(25, 120))


if __name__ == "__main__":
    main()
    #doc = nlp("This is a test sentence.")
    #for token in doc:
    #    print(token, token.pos_)
