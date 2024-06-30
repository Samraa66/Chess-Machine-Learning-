import math
import numpy as np
class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken # see if we can convert this to e4 type for format
        self.prior = prior # this is the probability that was generated based on the model

        self.children = [] # can we convert printing this to

        self.visit_count = 0
        self.value_sum = 0

    def is_fully_expanded(self):
        return len(self.children) > 0

    def select(self):
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb

        return best_child

    def get_ucb(self, child):
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2                         # using policy from model
        return q_value + self.args['C'] * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior

    def expand(self, policy):
        # NEED TO FIGURE OUT HOW TO ITERATE THROUGH EACH MOVE AND ENCODE THE ACTIONS
        for action, prob in enumerate(policy):
            if prob > 0:
                # here is where we do the move and change player
                child_state = self.state.copy()# copying could require a lot of time
                child_state = self.game.get_next_state(child_state, action, 1) # here 1 is just the current player
                child_state = self.game.change_perspective(child_state, player=-1)

                child = Node(self.game, self.args, child_state, self, action, prob)
                self.children.append(child)

        #return child

    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1

        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)


class MCTS:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def search(self, state):
        root = Node(self.game, self.args, state)

        for search in range(self.args['num_searches']):
            node = root

            while node.is_fully_expanded():
                node = node.select()

            value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
            value = self.game.get_opponent_value(value) # just change to negative version of value

            if not is_terminal:
                # forward propogate using model
                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(node.state)).unsqueeze(0)
                )
                # softmax over all layers (I dont get why he softmaxes twice, maybe we could just do it over legal moves afterward)
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                # here is where we should only keep legal moves
                valid_moves = self.game.get_valid_moves(node.state)
                policy *= valid_moves
                # now softmax over legal moves
                policy /= np.sum(policy)

                value = value.item()

                node.expand(policy)

            node.backpropagate(value)

        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)

        # the structure of the visit count of MCTS (used for training)
        return action_probs