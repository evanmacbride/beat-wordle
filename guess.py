import re
import random
import pandas as pd
from collections import Counter
import math
from os import sys
import itertools
import numpy as np

class Guess:
	STARTERS = ['stoae', 'spear', 'slate', 'cares']
	VOWELS = ['e', 'a', 'o', 'i', 'u', 'y']

	def __init__(self, word_list, word=None):
		self.initial_word_list = word_list
		self.word_list = word_list
		self.strict_word_list = self.word_list
		self.current = word
		if self.current:
			self.guess_history = [self.current]
		else:
			self.guess_history = []
		self.soln_ltrs = set()
		self.soln_ltr_matches = [None] * Game.WORD_LEN

	def seen_complete_soln(self):
		seen = all(self.soln_ltr_matches)
		return seen

	def seen_soln_anagram(self):
		return len(list(self.soln_ltrs)) == Game.WORD_LEN

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

	def get_current_entropy(self):
		return self.get_word_entr_list()[0][1]

	def get_word_entr_list(self, strict=False):
		# Get the probability of each letter at each position
		if strict:
			using_word_list = self.strict_word_list
		else:
			using_word_list = self.word_list
		#wordgrid_df = pd.DataFrame([list(w) for w in using_word_list])
		wordgrid_df = pd.DataFrame([list(w) for w in self.strict_word_list])
		gridct_df = pd.DataFrame()
		for c in wordgrid_df:
			posnct = pd.DataFrame(dict(Counter(wordgrid_df.loc[:,c])),index=[c])
			gridct_df = pd.concat([gridct_df, posnct])
		gridct_df = gridct_df.fillna(0)
		# Get the entropy for each letter at each position
		ltrent_df = pd.DataFrame()
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
		# Get the total entropy for each word in word_list
		word_entrs = []
		strict_ltrs = []
		if strict:
			# Create set of letters found in the strict_word_list
			pass
		for w in using_word_list:
			sum = 0.0
			seen_this_word = set()
			for i,l in enumerate(w):
				if l == self.soln_ltr_matches[i]:
					continue
				# If l is not in strict_ltrs, sum += 0.0. We want to make sure we're
				# using our guesses to obtain new information about the words in
				# strict_word_list. If we try to fetch from ltrent_df a letter that
				# doesn't exist there, we'll throw a KeyError.
				try:
					if l in seen_this_word:
						sum += ltrent_df.loc[i,l] / 2
					else:
						sum += ltrent_df.loc[i,l]
				except KeyError:
					sum += 0.0
				seen_this_word.add(l)
			word_entrs.append((w,sum))
		word_entrs = sorted(word_entrs, key=lambda x: x[1], reverse=True)		
		return word_entrs

	def find_breaker(self, hard=False):
		ltrs = []
		'''
		# ORIGINAL only tracks single letters and ignores multiples of the same
		# letter within a word.
		for word in self.strict_word_list:
			seen_this_word = set(list(word))
			for ltr in seen_this_word:
				ltrs.append(ltr)
		'''
		# Count "tokens" instead of letters. Tokens capture when there are multiple
		# occurrences of a letter within a word.
		for word in self.strict_word_list:
			for i,ltr in enumerate(word):
				if list(word).count(ltr) > 1:
					ltrs.append(ltr + ltr)
				else:
					#ltrs.append(ltr + str(i))
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
			using_word_list = self.initial_word_list
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
			print("Considering:",breaker.upper())
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

	def __init__(self, fpath, sample=None):
		f = open(fpath,"r")
		all_words = f.readlines() 
		all_words = [w.strip() for w in all_words]
		trim_words = [w for w in all_words if not re.search(r'[^a-z]',w)]
		word_len_words = [w for w in trim_words if len(w) == self.WORD_LEN]
		if sample:
			self.word_list = random.sample(word_len_words, 4500)
		else:
			self.word_list = word_len_words
		random.shuffle(self.word_list)
		self.solution = random.choice(self.word_list)
		self.score_history = []
		self.won = False
		self.current_turn = 0

	def get_word_list(self,sample=None):
		if sample:
			words = random.sample(self.word_list, sample)
		else:
			words = self.word_list
		return words

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

# Run simulation to calculate win rate
max_simuls = 1000
games_won = 0
breaker_wins = []
breaker_losses = []
loss_solns = []
for i in range(max_simuls):
	game = Game("data/enable1.txt")
	game_words = game.get_word_list()
	true_soln = game.get_solution()
	#true_soln = 'goxes'
	#game.solution = true_soln
	guess = Guess(game_words)
	print(true_soln.upper())
	# game loop
	used_breaker = False
	breaker_last_turn = False
	breaker_info = None
	strict = False
	all_similar = False
	while not game.won and game.current_turn < Game.TURNS:
		turns_remaining = Game.TURNS - game.current_turn - 1
		# Check how much of the solution we have
		matched_ltrs = 0
		for l in guess.soln_ltr_matches:
			if l is not None and l == 2:
				matched_ltrs += 1
		if game.current_turn >= 2:
			all_similar =  guess.lookahead_similarity()
		if game.current_turn == 0:
			cur_guess = guess.add_manual_guess('slate')
			breaker_last_turn = False
		elif guess.seen_complete_soln():
			print("***STRICT MODE*** solution seen")
			strict = True
			cur_guess = guess.get_auto_guess(strict)
			breaker_last_turn = False
		elif len(guess.strict_word_list) <= turns_remaining + 1:
			print("***STRICT MODE*** few choices remain")
			strict = True
			cur_guess = guess.get_auto_guess(strict)
			breaker_last_turn = False
		elif game.current_turn == Game.TURNS - 1:
			print("***STRICT MODE*** final turn")
			strict = True
			cur_guess = guess.get_auto_guess(strict)
			breaker_last_turn = False
		elif len(guess.soln_ltrs) >= Game.WORD_LEN-1 or matched_ltrs >= Game.WORD_LEN-1 or all_similar:
			print("***BREAKER MODE*** only similar words remain")
			strict = True
			# Only log the word_list length for the first time find_breaker() is called.
			if not used_breaker:
				breaker_info = len(guess.strict_word_list)
			cur_guess = guess.find_breaker(hard=True)
			used_breaker = True
			breaker_last_turn = True
		elif game.current_turn >= Game.TURNS - 4 and len(guess.strict_word_list) > 4 and not breaker_last_turn:
			print("***BREAKER MODE*** too many words remain")
			strict = True
			if not used_breaker:
				breaker_info = len(guess.strict_word_list)
			cur_guess = guess.find_breaker(hard=True)
			used_breaker = True
			breaker_last_turn = True
		else:
			if strict:
				print("***STRICT MODE***")
			cur_guess = guess.get_auto_guess(strict)
			breaker_last_turn = False
		score = game.score_guess(cur_guess)
		print(turns_remaining, cur_guess.upper(), score, guess.strict_word_list[0:10],
					"...", len(guess.strict_word_list))
		if cur_guess == true_soln:
			print("YOU WIN")
			game.won = True
			games_won += 1
			if used_breaker:
				breaker_wins.append(breaker_info)
			break
		else:
			game.current_turn += 1
		guess.update(score)
	if not game.won:
		print("YOU LOSE")
		print("REMAINING WORDS: ", guess.strict_word_list)
		loss_solns.append(true_soln)
		if used_breaker:
			breaker_losses.append(breaker_info)
	print()
print(games_won, "/", max_simuls, "... WIN RATE:", games_won / max_simuls)
print("LOST ON:")
for s in loss_solns:
	print(s)
#print()
#print("BREAKER WINS", np.min(breaker_wins), np.max(breaker_wins), np.mean(breaker_wins), np.median(breaker_wins))
#print()
#print("BREAKER LOSSES", np.min(breaker_losses), np.max(breaker_losses), np.mean(breaker_losses), np.median(breaker_losses))