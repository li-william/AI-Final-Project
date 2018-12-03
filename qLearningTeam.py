from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys, math
from game import Directions
import game
from util import nearestPoint
from game import Actions

#################
# Team creation #
#################

DEBUG = False 
interestingValues = {}

def createTeam(firstIndex, secondIndex, isRed,
				first = 'ApproximateQAgent', second = 'ApproximateQAgent', **args):
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
	if 'numTraining' in args:
		interestingValues['numTraining'] = args['numTraining']
	return [eval(first)(firstIndex), eval(second)(secondIndex)]

class ApproximateQAgent(CaptureAgent):

	def __init__( self, index ):
		CaptureAgent.__init__(self, index)
		self.weights = util.Counter()
		self.numTraining = 0
		if 'numTraining' in interestingValues:
			self.numTraining = interestingValues['numTraining']
		self.episodesSoFar = 0
		self.epsilon = 0.15
		self.discount = 0.8
		self.alpha = 0.3

	def registerInitialState(self, gameState):
		self.start = gameState.getAgentPosition(self.index)
		self.lastAction = None
		CaptureAgent.registerInitialState(self, gameState)

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

	def chooseAction(self, state):
		# Append game state to observation history...
		self.observationHistory.append(state)
		# Pick Action
		legalActions = state.getLegalActions(self.index)
		action = None
		if (DEBUG):
			print self.newline()
			print "AGENT " + str(self.index) + " choosing action!"
		if len(legalActions):
			if util.flipCoin(self.epsilon) and self.isTraining():
				action = random.choice(legalActions)
				if (DEBUG):
					print "ACTION CHOSE FROM RANDOM: " + action
			else:
				action = self.computeActionFromQValues(state)
				if (DEBUG):
					print "ACTION CHOSE FROM Q VALUES: " + action

		self.lastAction = action
		"""
		TODO
		ReflexCaptureAgent has some code that returns to your side if there are less than 2 pellets
		We added that here
		"""
		foodLeft = len(self.getFood(state).asList())

		if foodLeft <= 2:
			bestDist = 9999
			for a in legalActions:
				successor = self.getSuccessor(state, a)
				pos2 = successor.getAgentPosition(self.index)
				dist = self.getMazeDistance(self.start,pos2)
				if dist < bestDist:
					action = a
					bestDist = dist

		if (DEBUG):
			print "AGENT " + str(self.index) + " chose action " + action + "!"
		return action

	def getFeatures(self, gameState, action):
		"""
		Returns a counter of features for the state
		"""
		features = util.Counter()
		# Compute score from successor state
		successor = self.getSuccessor(gameState,action)
		features['successorScore'] = self.getScore(successor)-self.getScore(gameState)

		# get current position of the agent
		myPos = gameState.getAgentState(self.index).getPosition()
		
		#get next position of the agent
		sucPos = successor.getAgentState(self.index).getPosition()

		# Compute distance to the nearest food
		foodList = self.getFood(gameState).asList()
		if len(foodList) > 0:
			minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
			sucFoodDistance = min([self.getMazeDistance(sucPos, food) for food in foodList])
			features['towardsFood'] = minDistance-sucFoodDistance

		# Compute distance to closest ghost
		features['towardsGhost'] = 0
		enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
		inRange = filter(lambda x: not x.isPacman and x.getPosition() != None and self.getMazeDistance(myPos,x.getPosition())<5 and x.scaredTimer<5, enemies)
		if len(inRange) > 0:
			positions = [agent.getPosition() for agent in inRange]
			closest = min(positions, key=lambda x: self.getMazeDistance(myPos, x))
			ghostDis = self.getMazeDistance(myPos, closest)
			sucGhostDis=self.getMazeDistance(sucPos,closest)
			#the ghost is about to eat pacman
			if ghostDis<3:
				runaway=sucGhostDis-ghostDis
				#you are eaten
				if runaway>10:
					features['towardsGhost']=-100
				else:
					features['towardsGhost']=runaway
				capsuleList = self.getCapsules(gameState)
				if len(capsuleList) > 0:
					minCapDistance = min([self.getMazeDistance(myPos, c) for c in capsuleList])
					sucCapDistance=min([self.getMazeDistance(sucPos, c) for c in capsuleList])
					features['towardsCapsule'] = minCapDistance-sucCapDistance
				else:
					features['towardsCapsule'] = 0
		# Compute distance to the nearest capsule
		capsuleList = self.getCapsules(successor)
		if len(capsuleList) > 0:
			minDistance = min([self.getMazeDistance(myPos, c) for c in capsuleList])
			features['distanceToCapsule'] = minDistance
		else:
			features['distanceToCapsule'] = 0

		# Number of food currently carrying.	
		features['carrying'] = successor.getAgentState(self.index).numCarrying
		return features

	def getWeights(self):
		# Most positive = best choice
		return {
			'successorScore': 500,
			'towardsFood': 200, 
			'towardsGhost': 500, 
			'towardsCapsule': 300, 
			'distanceToCapsule': -300,
			'carrying': 100
			}

	def getQValue(self, state, action):
		"""
		  Should return Q(state,action) = w * featureVector
		  where * is the dotProduct operator
		"""
		total = 0
		weights = self.getWeights()
		features = self.getFeatures(state, action)
		for feature in features:
			# Implements the Q calculation
			total += features[feature] * weights[feature]
		return total

	def computeActionFromQValues(self, state):
		"""
		  Compute the best action to take in a state.  Note that if there
		  are no legal actions, which is the case at the terminal state,
		  you should return None.
		"""
		bestValue = -999999
		bestActions = None
		for action in state.getLegalActions(self.index):
			# For each action, if that action is the best then
			# update bestValue and update bestActions to be
			# a list containing only that action.
			# If the action is tied for best, then add it to
			# the list of actions with the best value.
			value = self.getQValue(state, action)
			if (DEBUG):
				print "ACTION: " + action + "           QVALUE: " + str(value)
			if value > bestValue:
				bestActions = [action]
				bestValue = value
			elif value == bestValue:
				bestActions.append(action)
		if bestActions == None:
			return Directions.STOP # If no legal actions return None
		return random.choice(bestActions) # Else choose one of the best actions randomly


	def computeValueFromQValues(self, state):
		"""
		  Returns max_action Q(state,action)
		  where the max is over legal actions.  Note that if
		  there are no legal actions, which is the case at the
		  terminal state, you should return a value of 0.0.
		"""
		bestValue = -999999
		noLegalActions = True
		for action in state.getLegalActions(self.index):
			# For each action, if that action is the best then
			# update bestValue
			noLegalActions = False
			value = self.getQValue(state, action)
			if value > bestValue:
				bestValue = value
		if noLegalActions:
			return 0 # If there is no legal action return 0
		# Otherwise return the best value found
		return bestValue

	def getReward(self, gameState):
		foodList = self.getFood(gameState).asList()
		return -len(foodList)

	def observationFunction(self, gameState):
		if len(self.observationHistory) > 0 and self.isTraining():
			self.update(self.getCurrentObservation(), self.lastAction, gameState, self.getReward(gameState))
		return gameState.makeObservation(self.index)

	def isTraining(self):
		return self.episodesSoFar < self.numTraining

	def update(self, state, action, nextState, reward):
		"""
		   Should update your weights based on transition
		"""
		if (DEBUG):
			print self.newline()
			print "AGENT " + str(self.index) + " updating weights!"
			print "Q VALUE FOR NEXT STATE: " + str(self.computeValueFromQValues(nextState))
			print "Q VALUE FOR CURRENT STATE: " + str(self.getQValue(state, action))
		difference = (reward + self.discount * self.computeValueFromQValues(nextState))
		difference -= self.getQValue(state, action)
		# Only calculate the difference once, not in the loop.
		newWeights = self.weights.copy()
		# Same with weights and features.
		features = self.getFeatures(state, action)
		for feature in features:
			# Implements the weight updating calculations
			newWeight = newWeights[feature] + self.alpha * difference * features[feature]
			if (DEBUG):
				print "AGENT " + str(self.index) + " weights for " + feature + ": " + str(newWeights[feature]) + " ---> " + str(newWeight)
			newWeights[feature]  = newWeight
		self.weights = newWeights.copy()
		# print "WEIGHTS AFTER UPDATE"
		# print self.weights

	def newline(self):
		return "-------------------------------------------------------------------------"

	def final(self, state):
		"Called at the end of each game."
		# call the super-class final method
		CaptureAgent.final(self, state)
		if self.isTraining() and DEBUG:
			print "END WEIGHTS"
			print self.weights
		self.episodesSoFar += 1
		if self.episodesSoFar == self.numTraining:
			print "FINISHED TRAINING"