import re
import random
import pandas as pd
from collections import Counter
import math
from os import sys

class Guess:
	STARTERS = ['stoae', 'spear', 'slate', 'cares', 'crane']
	VOWELS = ['e', 'a', 'o', 'i', 'u', 'y']

	def __init__(self, word_list, aux_word_list=None):
		self.initial_word_list = word_list
		self.word_list = word_list
		self.strict_word_list = self.word_list
		if aux_word_list:
			self.aux_word_list = aux_word_list
		else:
			self.aux_word_list = self.initial_word_list
		self.current = None
		if self.current:
			self.guess_history = [self.current]
		else:
			self.guess_history = []
		self.soln_ltrs = set()
		self.soln_ltr_matches = [None] * Game.WORD_LEN

	def seen_complete_soln(self):
		seen = all(self.soln_ltr_matches)
		return seen

	def add_manual_guess(self, word):
		self.current = word
		self.guess_history.append(self.current)
		return self.current

	def get_guess_history(self):
		return self.guess_history

	def get_current(self):
		return self.current

	def get_auto_guess(self, strict=False):
		self.current = self.get_word_entr_list(strict)[0][0]
		self.guess_history.append(self.current)
		return self.current

	def get_word_entr_list(self, strict=False):
		# DEBUG
		strict = True
		# Get the probability of each letter at each position or within entire word
		if strict:
			using_word_list = self.strict_word_list
		else:
			using_word_list = self.word_list
		wordgrid_df = pd.DataFrame([list(w) for w in self.strict_word_list])
		gridct_df = pd.DataFrame()
		# For each letter position, count the occurrences of letters at that 
		# position. Store this information in gridct_df.
		for c in wordgrid_df:
			posnct = pd.DataFrame(dict(Counter(wordgrid_df.loc[:,c])),index=[c])
			if strict and self.soln_ltr_matches[c]:
				continue
			gridct_df = pd.concat([gridct_df, posnct])
		gridct_df = gridct_df.fillna(0)
		if strict:
			sumgrid_df = gridct_df.sum(numeric_only=True, axis=0)
		else:
			sumgrid_df = None
		# Get the entropy for each letter at each position
		ltrent_df = pd.DataFrame()
		if not strict:
			# Use positional information
			for posn in wordgrid_df:
				posn_entr = {}
				for col in gridct_df:
					prob = gridct_df.loc[posn,col] / len(wordgrid_df)
					try:
						entr = -prob * math.log(prob, 2)
					except ValueError:
						entr = 0.0
					posn_entr[col] = entr
				ltrent_df = pd.concat([ltrent_df,pd.DataFrame(posn_entr,index=[posn])],axis=0)
		else:
			# Do not use positional information. Consider each letter's probability
			# at any position within a word.
			ltr_entr = {}
			for idx,ltrct in zip(sumgrid_df.index, sumgrid_df):
				prob = ltrct / (len(using_word_list) * len(using_word_list[0]))
				try:
					entr = -prob * math.log(prob, 2)
				except ValueError:
					entr = 0.0
				ltr_entr[idx] = entr
			ltrent_df = pd.Series(ltr_entr).to_frame()
		# Get the total entropy for each word in word_list
		word_entrs = []
		strict_ltrs = []
		for w in using_word_list:
			sum = 0.0
			seen_this_word = set()
			for i,l in enumerate(w):
				if l == self.soln_ltr_matches[i]:
					continue
				# If l is not in strict_ltrs, sum += 0.0. We want to make sure we're
				# using our guesses to obtain new information about the words in
				# strict_word_list.
				if not strict:
					try:
						addend = ltrent_df.loc[i,l]
						if l in seen_this_word:
							addend /= 2.0
						sum += addend
					# If we try to fetch from ltrent_df a letter that doesn't exist
					# in ltrent_df, we'll throw a KeyError.
					except KeyError:
						sum += 0.0
				else:
					sum += ltrent_df.loc[l].item()
				seen_this_word.add(l)
			word_entrs.append((w,sum))
		word_entrs = sorted(word_entrs, key=lambda x: x[1], reverse=True)		
		return word_entrs

	def find_breaker(self, hard=False):
		ltrs = []
		# Count "tokens" instead of letters. Tokens capture when there are multiple
		# occurrences of a letter within a word.
		for word in self.strict_word_list:
			for i,ltr in enumerate(word):
				if list(word).count(ltr) > 1:
					ltrs.append(ltr + ltr)
				else:
					ltrs.append(ltr)
		ltrcts = dict(Counter(ltrs))
		# Give a higher score to letters that are closer to being in half the words
		# in strict_word_list.
		ltrpts = []
		mid = len(self.strict_word_list) / 2.0
		for l,c in ltrcts.items():
			if c == len(self.strict_word_list):
				continue
			else:
				if c == mid:
					pts = 4.0
				else:
					pts = 1/abs(c-mid)
				# We want to get as much new information as possible
				if l in self.soln_ltr_matches:
					pts = 0.0
				elif l in self.soln_ltrs:
					pts = pts / 2.0
				ltrpts.append((l,pts))
		ltrpt_dict = dict(ltrpts)
		# Assign scores to words in the initial_word_list
		breaker_word_pts = []
		if hard:
			using_word_list = self.aux_word_list
		else:
			using_word_list = self.word_list
		for word in using_word_list:
			word_pts = 0.0
			seen_this_word = set()
			for l in word:
				if l in seen_this_word:
					continue
				else:
					if ltrpt_dict.get(l):
						word_pts += ltrpt_dict.get(l)
						seen_this_word.add(l)
			breaker_word_pts.append((word,word_pts))
		breaker_word_pts = sorted(breaker_word_pts, key=lambda x: x[1], reverse=True)
		redundant = True
		i = 0
		while redundant:
			breaker = breaker_word_pts[i][0]
			i += 1
			if breaker in self.guess_history:
				continue
			all_new_info = True
			for j,l in enumerate(breaker):
				if l == self.soln_ltr_matches[j]:
					all_new_info = False
					break
			if all_new_info:
				break
		self.current = breaker
		self.guess_history.append(self.current)
		return self.current

	def update(self, score):
		self.trim_word_lists(score)
		self.save_soln_ltrs(score)

	def save_soln_ltrs(self, score):
		for i, point in enumerate(score):
			if point != 0:
				self.soln_ltrs.add(self.current[i])
			if point == 2:
				self.soln_ltr_matches[i] = self.current[i]

	def trim_word_lists(self, score):
		self.trim_word_list(score)
		self.trim_strict_word_list(score)

	def trim_word_list(self, score):
		if self.current in self.word_list:
			self.word_list.remove(self.current)
		for i, point in enumerate(score):
			if point == 0:
				self.word_list = [w for w in self.word_list if self.current[i] not in w]
			elif point == 1:
				self.word_list = [w for w in self.word_list if self.current[i] in w 
													and self.current[i] != w[i]]
			elif point == 2:
				self.word_list = [w for w in self.word_list if self.current[i] in w]

	def trim_strict_word_list(self, score):
		if self.current in self.strict_word_list:
			self.strict_word_list.remove(self.current)
		for i, point in enumerate(score):
			if point == 0:
				self.strict_word_list = [w for w in self.strict_word_list if self.current[i] not in w]
			elif point == 1:
				self.strict_word_list = [w for w in self.strict_word_list if self.current[i] in w 
													and self.current[i] != w[i]]
			elif point == 2:
				self.strict_word_list = [w for w in self.strict_word_list if self.current[i] == w[i]]

	def lookahead_similarity(self):
		all_similar = False
		diff_ltrs = 0
		check_word = self.strict_word_list[0]
		for i in range(len(check_word)):
			for word in self.strict_word_list:
				if check_word[i] != word[i]:
					diff_ltrs += 1
					break
			if diff_ltrs >= 2:
				break
		if diff_ltrs <= 1:
			all_similar = True
		return all_similar

class Game:
	WORD_LEN = 5
	TURNS = 6
	NUM_DISP = 10

	def __init__(self, fpath, aux_fpath=None, sample=None):
		f = open(fpath,"r")
		all_words = f.readlines() 
		all_words = [w.strip() for w in all_words]
		trim_words = [w for w in all_words if not re.search(r'[^a-z]',w)]
		word_len_words = [w for w in trim_words if len(w) == self.WORD_LEN]
		if sample:
			self.word_list = random.sample(word_len_words, sample)
		else:
			self.word_list = word_len_words
		self.reset()
		f.close()
		if aux_fpath:
			f = open(aux_fpath,"r")
			all_aux_words = f.readlines() 
			all_aux_words = [w.strip() for w in all_aux_words]
			trim_aux_words = [w for w in all_aux_words if not re.search(r'[^a-z]',w)]
			word_len_aux_words = [w for w in trim_aux_words if len(w) == self.WORD_LEN]
			self.aux_word_list = word_len_aux_words
			random.shuffle(self.aux_word_list)
			f.close()
		else:
			self.aux_word_list = None
		return

	def reset(self):
		random.shuffle(self.word_list)
		self.solution = random.choice(self.word_list)
		self.score_history = []
		self.won = False
		self.current_turn = 0
		return

	def get_word_list(self,sample=None):
		if sample:
			words = random.sample(self.word_list, sample)
		else:
			words = self.word_list
		return words

	def get_aux_word_list(self, sample=None):
		return self.aux_word_list

	def get_solution(self):
		return self.solution 

	def score_guess(self, guess):
		score = []
		for g,s in zip(guess,self.solution):
			if g == s:
				score.append(2)
			elif g in self.solution:
				score.append(1)
			else:
				score.append(0)
		self.score_history.append(score)
		return score

	def get_clr_scored_guess(self, score, guess):
		'''
		Return a string formatted to color-code a guess's score when printed in
		the terminal. A zero-scoring letter (corresponding to that letter not being
		part of the true solution) should appear darkest. A one score should appear
		lighter than a zero score, but not very bright. A two score should appear
		the brightest. Different systems may display colors differently. You should
		adjust the values of the score_ variables to suit your needs.
		'''
		guess = guess.upper()
		score_0 = 31
		score_1 = 34
		score_2 = 36
		clr = 0
		clr_scored_guess = ''
		fmt_str = "\x1b[{}m{}\x1b[0m"
		for i, s in enumerate(score):
			if s == 0:
				clr = score_0
			elif s == 1:
				clr = score_1
			elif s == 2:
				clr = score_2
			else:
				clr = 0
			clr_scored_guess += fmt_str.format(clr,guess[i])
		clr_scored_guess = clr_scored_guess
		return clr_scored_guess

	def get_display_header(self, verbose=True):
		if verbose:
			header_str = (
'''T  GUESS  SCORE           REMAINING                                                               SIZE
_  _____  ______________  _________                                                               ____
			''')
		else:
			header_str = (
'''T  GUESS  SCORE
_  _____  ______________
			''')
		return header_str

def simulation(num_simuls=1, verbose=True, manual_soln=None, starter='slate'):
	# Run simulation to calculate win rate
	games_won = 0
	loss_solns = []
	turns_played = 0
	# Initialize game
	game = Game("data/popular.txt", aux_fpath="data/enable1.txt")
	header_str = game.get_display_header(verbose)
	if not verbose:
		print(header_str)
	for i in range(num_simuls):
		# Initialize guess engine
		game_words = game.get_word_list()
		aux_words = game.get_aux_word_list()
		if not manual_soln:
			true_soln = game.get_solution()
		else:
			true_soln = manual_soln
			game.solution = true_soln
		guess = Guess(game_words, aux_words)
		# Gameplay booleans
		used_breaker = False
		breaker_last_turn = False
		strict = False
		all_similar = False
		if verbose:
			print(header_str)
		# Game loop
		while not game.won and game.current_turn < Game.TURNS:
			turns_remaining = Game.TURNS - game.current_turn - 1
			# Check how much of the solution we have
			matched_ltrs = 0
			for l in guess.soln_ltr_matches:
				if l is not None and l == 2:
					matched_ltrs += 1
			if game.current_turn >= 2:
				all_similar =  guess.lookahead_similarity()
			if game.current_turn == 0 and starter:
				cur_guess = guess.add_manual_guess(starter)
				breaker_last_turn = False
			elif guess.seen_complete_soln():
				if verbose:
					print("***STRICT MODE*** solution seen")
				strict = True
				cur_guess = guess.get_auto_guess(strict)
				breaker_last_turn = False
			elif len(guess.strict_word_list) <= turns_remaining + 1:
				if verbose:
					print("***STRICT MODE*** few choices remain")
				strict = True
				cur_guess = guess.get_auto_guess(strict)
				breaker_last_turn = False
			elif game.current_turn == Game.TURNS - 1:
				if verbose:
					print("***STRICT MODE*** final turn")
				strict = True
				cur_guess = guess.get_auto_guess(strict)
				breaker_last_turn = False
			elif len(guess.soln_ltrs) >= Game.WORD_LEN-1 or matched_ltrs >= Game.WORD_LEN-1 or all_similar:
				if verbose:
					print("***BREAKER MODE*** only similar words remain")
				strict = True
				cur_guess = guess.find_breaker(hard=True)
				breaker_last_turn = True
			elif game.current_turn >= Game.TURNS - 4 and len(guess.strict_word_list) > (
						Game.TURNS - game.current_turn + 1) and ( 
					not breaker_last_turn):
				if verbose:
					print("***BREAKER MODE*** too many words remain")
				strict = True
				cur_guess = guess.find_breaker(hard=True)
				breaker_last_turn = True
			elif game.current_turn >= 2:
				if verbose:
					print("***STRICT MODE*** opening moves have ended")
				strict = True
				cur_guess = guess.get_auto_guess(strict)
				breaker_last_turn = False
			else:
				if strict and verbose:
					print("***STRICT MODE***")
				cur_guess = guess.get_auto_guess(strict)
				breaker_last_turn = False
			score = game.score_guess(cur_guess)
			clr_scored_guess = game.get_clr_scored_guess(score, cur_guess)
			if verbose:
				display_word_str = ", ".join(guess.strict_word_list[0:game.NUM_DISP])
				# Only print the first NUM_DISP words from the word list
				if len(guess.strict_word_list) > game.NUM_DISP:
					display_word_str += "... " + '{0:>4}'.format(str(len(guess.strict_word_list)))
				else:
					display_word_str = '{0:<75}'.format(display_word_str) + str(
						len(guess.strict_word_list))
				display_word_str = " " + display_word_str
				print("{}:".format(turns_remaining), clr_scored_guess, score, display_word_str)
			else:
				print("{}:".format(turns_remaining), clr_scored_guess, score)
			if cur_guess == true_soln:
				print("YOU WIN")
				game.won = True
				games_won += 1
				break
			else:
				game.current_turn += 1
			guess.update(score)
		if not game.won:
			print("YOU LOSE")
			print("SOLN WAS:", true_soln.upper())
			print("Remaining words:") 
			for word in guess.strict_word_list:
				print(word)
			loss_solns.append(true_soln)
		# Reset game to get new solution etc. for next turn
		turns_played += game.current_turn + 1
		game.reset()
		print()
	print(games_won, "/", num_simuls, "... WIN RATE:", games_won / num_simuls, 
				"  AVG TURNS PLAYED:", turns_played / num_simuls)
	if num_simuls != games_won:
		print("LOST ON:")
		for s in loss_solns:
			print(s)
	return

if __name__ == '__main__':
	# Arguments and options for simulation()
	verbose = True
	num_simuls = 1
	manual_soln = None
	starter = 'slate'
	# Parse command line input
	opts = [opt for opt in sys.argv if "--" in opt]
	args = [arg for arg in sys.argv if "-" in arg and "--" not in arg]
	# Parse options
	if "--quiet" in opts:
		verbose = False
	elif "--verbose" in opts:
		verbose = True
	# Parse arguments
	for arg in args:
		if "-n=" in arg:
			num_simuls = int(arg.split("=")[1])
		if "-m=" in arg:
			manual_soln = arg.split("=")[1].lower()
		if "-s=" in arg:
			value = arg.split("=")[1]
			if value == "None":
				starter = None
			else:
				starter = arg.split("=")[1].lower()
	sys.exit(simulation(verbose=verbose, num_simuls=num_simuls, manual_soln=manual_soln, starter=starter))
