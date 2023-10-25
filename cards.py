import enum
import random

ranks = {
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "jack": 10,
    "queen": 10,
    "king": 10,
    "ace": (1,11)
}

class Suits(enum.Enum):
    spades = "spades"
    clubs = "clubs"
    hearts = "hearts"
    diamonds = "diamonds"

class Cards:
    def __init__(self, rank, suit, value):
        self.suit = suit,
        self.rank = rank,
        self.value = value
        
    def __str__(self):
        return self.rank + "of " + self.suit.value
    
class Deck:
    def __init__(self, num=2):
        self.cards=[]
        for i in range(num):
            for suit in Suits:
                for rank, value in ranks.items():
                    self.cards.append(Cards(suit, rank, value))
                    
    def shuffle_deck(self):
        random.shuffle(self.cards)
        
    def deal_cards(self):
        return self.cards.pop(0)
    
    def peek(self):
        if len(self.cards) > 0:
            return self.cards[0]
    
    