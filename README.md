# Blackjack_AI

## Goals:
  Create a machine learning model that uses Q-learning and reinforcement learning to play Blackjack.
  Implement card counting to affect how the model makes decisions on moves using standard card counting strategies
    -- if card value is 2-6, subtract 1
    -- if card value is 7-9, add 0
    -- if card value is 10-Ace, add 1

## Environment:
  The game is set up using OpenAI Gym. Within the program, a BlackjackAgent class is defined which allows manipulation of the enviornment such as
  getting each action, checking game status, etc.
