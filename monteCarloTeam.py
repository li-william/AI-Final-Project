from captureAgents import CaptureAgent,ReflexCaptureAgent,OffensiveReflexAgent,DefensiveReflexAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint

def createTeam(firstIndex, secondIndex, isRed,
               first = 'monteCarloAttacker', second = 'DefensiveReflexAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

class monteCarloAttacker(OffensiveReflexAgent):
    
    def randomSimulation(self, depth, gameState, decay):
        """
        Random simulate some actions for the agent. The actions other agents can take
        are ignored, or, in other words, we consider their actions is always STOP.
        The final state from the simulation is evaluated. Decay is to ensure that
        the importance of results lowers with depth.
        """
        copy = gameState.deepCopy()
        value = self.evaluate(copy, Directions.STOP)
        decay_index = 1

        while depth:
            actions = copy.getLegalActions(self.index)
            # The agent should not stay put in the simulation
            current_direction = copy.getAgentState(self.index).configuration.direction
            # The agent should not use the reverse direction during simulation

            reversed_direction = Directions.REVERSE[copy.getAgentState(self.index).configuration.direction]
            if reversed_direction in actions and len(actions) > 1:
                actions.remove(reversed_direction)
            # Randomly chooses a valid action
            a = random.choice(actions)
            copy = copy.generateSuccessor(self.index, a)
            value += decay ** decay_index * self.evaluate(copy, Directions.STOP)
            depth -= 1
            decay_index += 1

        return value

    def chooseAction(self, gameState):
        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()

        # Get valid actions. Randomly choose a valid one out of the best (if best is more than one)
        actions = gameState.getLegalActions(self.index)
        actions.remove(Directions.STOP)
        possibleValue = []

        for a in actions:
            next_state = gameState.generateSuccessor(self.index, a)
            score = 0
            for i in range(0, 10):
                score += self.randomSimulation(1, next_state, 0.8) / 10
            possibleValue.append(score)

        bestAction = max(possibleValue)
        possibleChoice = filter(lambda x: x[0] == bestAction, zip(possibleValue, actions))
        # print 'eval time for offensive agent %d: %.4f' % (self.agent.index, time.time() - start)
        return random.choice(possibleChoice)[1]
