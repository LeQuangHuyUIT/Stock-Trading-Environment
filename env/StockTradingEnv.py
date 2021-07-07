#================================================================
#
#   File name   : RL-Bitcoin-trading-bot_2.py
#   Author      : PyLessons
#   Created date: 2020-12-12
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/RL-Bitcoin-trading-bot
#   Description : Trading Crypto with Reinforcement Learning #2
#
#================================================================
import pandas as pd
import numpy as np
import random
from collections import deque
# in colab
from env.utils import TradingGraph, Write_to_file
# from utils import TradingGraph, Write_to_file
class CustomEnv:
	# A custom Bitcoin trading environment
	def __init__(self, df, initial_balance=1000, lookback_window_size=50, Render_range=100, Show_reward=False, Show_indicators=False, normalize_value=40000):
		# Define action space and state size and other custom parameters
		self.df = df.dropna().reset_index()
		self.df_total_steps = len(self.df)-1
		self.initial_balance = initial_balance
		self.lookback_window_size = lookback_window_size
		self.Render_range = Render_range # render range in visualization
		self.Show_reward = Show_reward # show order reward in rendered visualization
		self.Show_indicators = Show_indicators # show main indicators in rendered visualization

		# Orders history contains the balance, net_worth, crypto_bought, crypto_sold, crypto_held values for the last lookback_window_size steps
		self.orders_history = deque(maxlen=self.lookback_window_size)
		
		# Market history contains the OHCL values for the last lookback_window_size prices
		self.market_history = deque(maxlen=self.lookback_window_size)

		self.indicators_history = deque(maxlen=self.lookback_window_size)

		self.normalize_value = normalize_value

	# Reset the state of the environment to an initial state
	def reset(self, env_steps_size = 0):
		self.visualization = TradingGraph(Render_range=self.Render_range, Show_reward=self.Show_reward, Show_indicators=self.Show_indicators) # init visualization
		self.trades = deque(maxlen=self.Render_range) # limited orders memory for visualization
		
		self.balance = self.initial_balance
		self.net_worth = self.initial_balance
		self.prev_net_worth = self.initial_balance
		self.crypto_held = 0
		self.crypto_sold = 0
		self.crypto_bought = 0
		self.episode_orders = 0 # track episode orders count
		self.prev_episode_orders = 0 # track previous episode orders count
		self.rewards = deque(maxlen=self.Render_range)
		self.env_steps_size = env_steps_size
		self.punish_value = 0
		if env_steps_size > 0: # used for training dataset
			self.start_step = random.randint(self.lookback_window_size, self.df_total_steps - env_steps_size)
			self.end_step = self.start_step + env_steps_size
		else: # used for testing dataset
			self.start_step = self.lookback_window_size
			self.end_step = self.df_total_steps
			
		self.current_step = self.start_step

		for i in reversed(range(self.lookback_window_size)):
			current_step = self.current_step - i
			self.orders_history.append([self.balance, self.net_worth, self.crypto_bought, self.crypto_sold, self.crypto_held])

			self.market_history.append([self.df.loc[current_step, 'Open'],
										self.df.loc[current_step, 'High'],
										self.df.loc[current_step, 'Low'],
										self.df.loc[current_step, 'Close'],
										self.df.loc[current_step, 'Volume'],
										])

			self.indicators_history.append(
								[self.df.loc[current_step, 'sma7'],
								self.df.loc[current_step, 'trend_adx'],
								self.df.loc[current_step, 'trend_adx_pos'],
								self.df.loc[current_step, 'trend_adx_neg'],
								self.df.loc[current_step, 'trend_psar_up_indicator'],
								self.df.loc[current_step, 'trend_psar_down_indicator'],
								self.df.loc[current_step, 'volatility_atr'],
								self.df.loc[current_step, 'volatility_bbli'],
								self.df.loc[current_step, 'others_dr'],
								self.df.loc[current_step, 'others_cr']
										])
			

		state = np.concatenate((self.market_history, self.orders_history), axis=1) / self.normalize_value
		state = np.concatenate((state, self.indicators_history), axis=1)

		return state

	# Get the data points for the given current_step
	def _next_observation(self):
		self.market_history.append([self.df.loc[self.current_step, 'Open'],
									self.df.loc[self.current_step, 'High'],
									self.df.loc[self.current_step, 'Low'],
									self.df.loc[self.current_step, 'Close'],
									self.df.loc[self.current_step, 'Volume'],
									])

		self.indicators_history.append([
									self.df.loc[self.current_step, 'sma7'],
									self.df.loc[self.current_step, 'trend_adx'],
									self.df.loc[self.current_step, 'trend_adx_pos'],
									self.df.loc[self.current_step, 'trend_adx_neg'],
									self.df.loc[self.current_step, 'trend_psar_up_indicator'],
									self.df.loc[self.current_step, 'trend_psar_down_indicator'],
									self.df.loc[self.current_step, 'volatility_atr'],
									self.df.loc[self.current_step, 'volatility_bbli'],
									self.df.loc[self.current_step, 'others_dr'],
									self.df.loc[self.current_step, 'others_cr']
									])
		
		obs = np.concatenate((self.market_history, self.orders_history), axis=1) / self.normalize_value
		obs = np.concatenate((obs, self.indicators_history), axis=1)
		
		return obs

	# Execute one time step within the environment
	def step(self, action):
		self.crypto_bought = 0
		self.crypto_sold = 0
		self.current_step += 1

		# Set the current price to a random price between open and close
		#current_price = random.uniform(
		#    self.df.loc[self.current_step, 'Open'],
		#    self.df.loc[self.current_step, 'Close'])
		current_price = self.df.loc[self.current_step, 'Open']
		Date = self.df.loc[self.current_step, 'Date'] # for visualization
		High = self.df.loc[self.current_step, 'High'] # for visualization
		Low = self.df.loc[self.current_step, 'Low'] # for visualization

		if action == 0: # Hold
			pass

		elif action == 1 and self.balance > self.initial_balance/100:
			# Buy with 100% of current balance
			self.crypto_bought = self.balance / current_price
			self.balance -= self.crypto_bought * current_price
			self.crypto_held += self.crypto_bought
			self.trades.append({'Date' : Date, 'High' : High, 'Low' : Low, 'total': self.crypto_bought, 'type': "buy", 'current_price': current_price})
			self.episode_orders += 1

		elif action == 2 and self.crypto_held>0:
			# Sell 100% of current crypto held
			self.crypto_sold = self.crypto_held
			self.balance += self.crypto_sold * current_price
			self.crypto_held -= self.crypto_sold
			self.trades.append({'Date' : Date, 'High' : High, 'Low' : Low, 'total': self.crypto_sold, 'type': "sell", 'current_price': current_price})
			self.episode_orders += 1

		self.prev_net_worth = self.net_worth
		self.net_worth = self.balance + self.crypto_held * current_price

		self.orders_history.append([self.balance, self.net_worth, self.crypto_bought, self.crypto_sold, self.crypto_held])

		# Receive calculated reward
		reward = self.get_reward()

		if self.net_worth <= self.initial_balance/2:
			done = True
		else:
			done = False

		obs = self._next_observation()
		
		return obs, reward, done

	# Calculate reward
	def get_reward(self):
		self.punish_value += self.net_worth * 0.00001
		if self.episode_orders > 1 and self.episode_orders > self.prev_episode_orders:
			self.prev_episode_orders = self.episode_orders
			if self.trades[-1]['type'] == "buy" and self.trades[-2]['type'] == "sell":
				reward = self.trades[-2]['total']*self.trades[-2]['current_price'] - self.trades[-2]['total']*self.trades[-1]['current_price']
				reward -= self.punish_value
				self.punish_value = 0
				self.trades[-1]["Reward"] = reward
				return reward
			elif self.trades[-1]['type'] == "sell" and self.trades[-2]['type'] == "buy":
				reward = self.trades[-1]['total']*self.trades[-1]['current_price'] - self.trades[-2]['total']*self.trades[-2]['current_price']
				reward -= self.punish_value
				self.punish_value = 0
				self.trades[-1]["Reward"] = reward
				return reward
		else:
			return 0 - self.punish_value

	# render environment
	def render(self, visualize = False):
		#print(f'Step: {self.current_step}, Net Worth: {self.net_worth}')
		if visualize:
			# Render the environment to the screen
			img = self.visualization.render(self.df.loc[self.current_step], self.net_worth, self.trades)
			return img
