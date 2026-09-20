import math
import random
import time


class Nim:
    def _init_(self, initial=[1, 3, 5, 7]):
        self.piles = initial.copy()
        self.player = 0
        self.winner = None

    @classmethod
    def available_actions(cls, piles):
        actions = set()
        for i, pile in enumerate(piles):
            for j in range(1, pile + 1):
                actions.add((i, j))
        return actions

    @classmethod
    def other_player(cls, player):
        return 0 if player == 1 else 1

    def switch_player(self):
        self.player = Nim.other_player(self.player)

    def move(self, action):
        pile, count = action
        if self.winner is not None:
            raise Exception("Game already won")
        elif pile < 0 or pile >= len(self.piles):
            raise Exception("Invalid pile")
        elif count < 1 or count > self.piles[pile]:
            raise Exception("Invalid number of objects")
        self.piles[pile] -= count
        self.switch_player()
        if all(pile == 0 for pile in self.piles):
            self.winner = self.player


class NimAI:
    def __init__(self, alpha=0.5, epsilon=0.1):
        self.q = dict()
        self.alpha = alpha
        self.epsilon = epsilon

    def update(self, old_state, action, new_state, reward):
        old_q = self.get_q_value(old_state, action)
        future_reward = self.best_future_reward(new_state)
        self.update_q_value(old_state, action, old_q, reward, future_reward)

    def get_q_value(self, state, action):
        state_key = tuple(state)
        return self.q.get((state_key, action), 0)

    def update_q_value(self, state, action, old_q, reward, future_rewards):
        state_key = tuple(state)
        alpha = getattr(self, "alpha", 0.5)
        new_q_estimate = reward + future_rewards
        self.q[(state_key, action)] = old_q + alpha * (new_q_estimate - old_q)

    def best_future_reward(self, state):
        actions = Nim.available_actions(state)
        if not actions:
            return 0
        return max(self.get_q_value(state, act) for act in actions)

    def choose_action(self, state, epsilon=True):
        actions = list(Nim.available_actions(state))
        if not actions:
            raise Exception("No valid actions available in current state.")

        eps = getattr(self, "epsilon", 0.1) if epsilon else 0
        if epsilon and random.random() < eps:
            return random.choice(actions)

        action_scores = [(self.get_q_value(state, act), act) for act in actions]
        max_score = max(score for score, _ in action_scores)
        top_actions = [act for score, act in action_scores if score == max_score]

        return random.choice(top_actions)


def train(n):
    player = NimAI()

    for i in range(n):
        print(f"Playing training game {i + 1}")
        game = Nim()

        last_moves = {
            0: {"state": None, "action": None},
            1: {"state": None, "action": None}
        }

        while True:
            current_state = game.piles.copy()
            current_player = game.player
            chosen_action = player.choose_action(game.piles)

            last_moves[current_player]["state"] = current_state
            last_moves[current_player]["action"] = chosen_action

            game.move(chosen_action)
            next_state = game.piles.copy()

            if game.winner is not None:
                player.update(current_state, chosen_action, next_state, -1)
                winner_player = game.winner
                player.update(
                    last_moves[winner_player]["state"],
                    last_moves[winner_player]["action"],
                    next_state,
                    1
                )
                break
            elif last_moves[current_player]["state"] is not None:
                player.update(
                    last_moves[current_player]["state"],
                    last_moves[current_player]["action"],
                    next_state,
                    0
                )

    print("Done training")
    return player


def play(ai, human_player=None):
    if human_player is None:
        human_player = random.randint(0, 1)

    game = Nim()

    while True:
        print()
        print("Piles:")
        for i, pile in enumerate(game.piles):
            print(f"Pile {i}: {pile}")
        print()

        available_actions = Nim.available_actions(game.piles)
        time.sleep(1)

        if game.player == human_player:
            print("Your Turn")
            while True:
                pile = int(input("Choose Pile: "))
                count = int(input("Choose Count: "))
                if (pile, count) in available_actions:
                    break
                print("Invalid move, try again.")
        else:
            print("AI's Turn")
            pile, count = ai.choose_action(game.piles, epsilon=False)
            print(f"AI chose to take {count} from pile {pile}.")

        game.move((pile, count))

        if game.winner is not None:
            print()
            print("GAME OVER")
            winner = "Human" if game.winner == human_player else "AI"
            print(f"Winner is {winner}")
            return
