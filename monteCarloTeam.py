from captureAgents import CaptureAgent
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


class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """

  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

class monteCarloAttacker(ReflexCaptureAgent):
    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        # self.distancer.getMazeDistances()
        # self.retreat = False
        # self.numEnemyFood = "+inf"

        boundary = (gameState.data.layout.width - 2) / 2 if self.red else ((gameState.data.layout.width - 2) / 2) + 1

        self.boundary = []
        for i in range(1, gameState.data.layout.height - 1):
            if not gameState.hasWall(boundary, i):
                self.boundary.append((boundary, i))

        # Remove some positions. The agent do not need to patrol
        # all positions in the central area.
        # self.noWallSpots = []
        # while len(self.noWallSpots) > (gameState.data.layout.height - 2) / 2:
        #     self.noWallSpots.pop(0)
        #     self.noWallSpots.pop(len(self.noWallSpots) - 1)

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        features['successorScore'] = self.getScore(successor)
        currPos = successor.getAgentState(self.index).getPosition()

        # boundary distance
        min_boundary = min(self.getMazeDistance(currPos, bound) for bound in self.boundary)
        features['returned'] = min_boundary

        features['carrying'] = successor.getAgentState(self.index).numCarrying

        # food distance
        food_list = self.getFood(successor).asList()
        if len(food_list):
            min_food_dist = min(self.getMazeDistance(currPos, food) for food in food_list)
            features['distanceToFood'] = min_food_dist

        # capsule distance
        capsule_list = self.getCapsules(successor)
        if len(capsule_list) > 0:
            min_capsule_dist = min(self.getMazeDistance(currPos, cap) for cap in capsule_list)
            features['distanceToCapsule'] = min_capsule_dist
        else:
            features['distanceToCapsule'] = 0

        # ghost distance
        opponent_state = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        visible = filter(lambda x: not x.isPacman and x.getPosition() != None, opponent_state)
        if len(visible):
            positions = [agent.getPosition() for agent in visible]
            closest = min(positions, key=lambda x: self.getMazeDistance(currPos, x))
            closest_dist = self.getMazeDistance(currPos, closest)

            if closest_dist <= 5:
                features['GhostDistance'] = closest_dist
        else:
             probDist = []
             for i in self.getOpponents(successor):
                 probDist.append(successor.getAgentDistances()[i])
             features['GhostDistance'] = min(probDist)

        # Attacker only try to kill the enemy if : itself is ghost form and the distance between him and the ghost is less than 4
        enemies_vuln = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        visible_vuln = filter(lambda x: x.isPacman and x.getPosition() != None, enemies_vuln)
        if len(visible_vuln):
            positions = [agent.getPosition() for agent in visible_vuln]
            closest = min(positions, key=lambda x: self.getMazeDistance(currPos, x))
            closest_dist = self.getMazeDistance(currPos, closest)
            if closest_dist < 4:
                features['distanceToEnemiesPacMan'] = closest_dist
        else:
            features['distanceToEnemiesPacMan'] = 0

        return features


    def getWeights(self, gameState, action):
            """
            Get weights for the features used in the evaluation.
            """

            # If opponent is scared, the agent should not care about GhostDistance
            successor = self.getSuccessor(gameState, action)
            numOfFood = len(self.getFood(successor).asList())
            numOfCarrying = successor.getAgentState(self.index).numCarrying
            currPos = successor.getAgentState(self.index).getPosition()

            opponents = [successor.getAgentState(i) for i in self.getOpponents(successor)]
            visible = filter(lambda x: not x.isPacman and x.getPosition() != None, opponents)
            if len(visible):
                for agent in visible:
                    if agent.scaredTimer > 0:
                        if agent.scaredTimer > 12:
                            return {'successorScore': 110, 'distanceToFood': -10, 'distanceToEnemiesPacMan': 0,
                                    'GhostDistance': -1, 'distanceToCapsule': 0, 'carrying': 350, 'returned': 10-3*numOfCarrying }

                        elif 6 < agent.scaredTimer < 12 :
                            return {'successorScore': 110+5*numOfCarrying, 'distanceToFood': -5, 'distanceToEnemiesPacMan': 0,
                                    'GhostDistance': -15, 'distanceToCapsule': -10,'carrying': 100, 'returned': -5-4*numOfCarrying,
                                    }

                    # Visible and not scared
                    else:
                        return {'successorScore': 110, 'distanceToFood': -10, 'distanceToEnemiesPacMan': 0,
                                'GhostDistance': 20, 'distanceToCapsule': -15, 'carrying': 0, 'returned': -15,
                                }
                    # If I am not PacMan the enemy is a pacMan, I can try to eliminate him

            # Attacker only try to defence if it is close to it (less than 4 steps)
            enemies_vuln = [successor.getAgentState(i) for i in self.getOpponents(successor)]
            visible_vuln = filter(lambda x: x.isPacman and x.getPosition() != None, enemies_vuln)
            if len(visible_vuln) > 0 and not gameState.getAgentState(self.index).isPacman:
                return {'successorScore': 0, 'distanceToFood': -1, 'distanceToEnemiesPacMan': -8,
                        'distanceToCapsule': 0, 'GhostDistance': 0,
                        'returned': 0, 'carrying': 10}

            # Did not see anything
            return {'successorScore': 1000+numOfCarrying*3.5, 'distanceToFood': -7, 'GhostDistance': 0, 'distanceToEnemiesPacMan': 0,
                    'distanceToCapsule': -5,  'carrying': 350,'returned': 5-numOfCarrying*3}

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


class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}
