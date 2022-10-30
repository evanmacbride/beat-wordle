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
		self.current = self.get_most_likely_word()
		self.guess_history.append(self.current)
		return self.current

	def get_most_likely_word(self):
		all_ltrs = list(''.join(self.strict_word_list))
		ltrfreq = dict(Counter(all_ltrs))
		ltrentr = {}
		for ltr in ltrfreq.keys():
			prob = ltrfreq[ltr] / len(all_ltrs)
			if prob == 0:
				entr = 0
			else:
				entr = prob * -math.log(prob, 2)
			ltrentr[ltr] = entr
		wordentr = []
		for word in self.strict_word_list:
			entr_sum = 0.0
			seen_this_word = {}
			for ltr in word:
				div = 1
				if seen_this_word.get(ltr):
					div = seen_this_word[ltr]
					seen_this_word[ltr] = div + 1
				else:
					seen_this_word[ltr] = 2
				# Repeat letters contribute less to entr_sum
				entr_sum += ltrentr[ltr] / div
			wordentr.append((word, entr_sum))
		wordentr = sorted(wordentr, key=lambda x: x[1], reverse=True)
		most_likely_word = wordentr[0][0]
		return most_likely_word

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
		self.word_list = self.trim_word_list(score, self.word_list)
		self.strict_word_list = self.trim_word_list(score, self.strict_word_list)

	def trim_word_list(self, score, word_list):
		# TODO add a strict mode. Pass in a strict boolean as an argument.
		# Find the matches first
		exact_matches = set()
		# Keep track of how many partial matches there are for each letter so that
		# we can accurately trim our word_list.
		partial_matches = {}
		for i, point in enumerate(score):
			if point == 2:
				word_list = [w for w in word_list if self.current[i] == w[i]]
				exact_matches.add(self.current[i])
			elif point == 1:
				word_list = [w for w in word_list if self.current[i] in w]
				word_list = [w for w in word_list if self.current[i] != w[i]] # STRICT
				qty_found = partial_matches.get(self.current[i])
				if qty_found:
					partial_matches[self.current[i]] = qty_found + 1
				else:
					partial_matches[self.current[i]] = 1
		# Only remove words with zero-scoring letters if those letters weren't also
		# part of partial or exact matches elsewhere in the word.
		for i, point in enumerate(score):
			qty_found = partial_matches.get(self.current[i])
			if point == 0 and not (self.current[i] in exact_matches or qty_found):
				word_list = [w for w in word_list if self.current[i] not in w]
			elif point == 0 and self.current[i] in exact_matches:
				word_list = [w for w in word_list if self.current[i] != w[i]]
		return word_list
	
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
		self.reset()
		return

	def reset(self):
		random.shuffle(self.word_list)
		random.shuffle(self.aux_word_list)
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
		score = [0] * Game.WORD_LEN
		to_match_ct = {}
		for s in self.solution:
			to_match_ct[s] = self.solution.count(s)
		for i,(g,s) in enumerate(zip(guess, self.solution)):
			if g == s:
				score[i] = 2
				to_match_ct[g] -= 1
			else:
				if g in self.solution and to_match_ct[g] > 0:
					score[i] = 1
					to_match_ct[g] -= 1
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
	game = Game("data/popular_plus.txt", aux_fpath="data/enable1.txt")
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
				# FIRST TURN, USING STARTER
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
