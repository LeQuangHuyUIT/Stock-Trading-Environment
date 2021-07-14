#in colab
# env.StockTradingEnv2 import *
from StockTradingEnv2 import *
import cv2
import pandas as pd

lookback_window_size = 10
test_window = 60 # 60 turn 

# test with fpt
df_fpt = pd.read_csv('/home/huyle/MyGit/Stock-Trading-Environment/data/fpt_indicators.csv')
df_fpt['Date'] = df_fpt['Date'].apply(lambda x: datetime.strptime(str(x),'%Y%m%d'))
df_fpt = df_fpt.sort_values('Date')

train_df = df_fpt[:-test_window-lookback_window_size]
test_df = df_fpt[-test_window-lookback_window_size:]

test_env_fpt = CustomEnv(test_df, lookback_window_size= lookback_window_size)

agent_PPO_fpt = CustomAgent(lookback_window_size=lookback_window_size, lr=0.0001, epochs=1, optimizer=Adam, batch_size = 32, model="CNN")
agent_DQN_fpt = DQNAgent(test_env_fpt, lookback_window_size= lookback_window_size)
agent_PG_fpt = PGAgent("fpt_policy_gradient", 1e-5)

img_ppo = test_agent(test_env_fpt, agent_PPO_fpt ,test_episodes= 1, \
	folder= "/home/huyle/MyGit/Stock-Trading-Environment/weights",\
	name= "fpt_Crypto_trader" )
cv2.imwrite("fpt_PPO.jpg", img_ppo)

img_pg = test_agent(test_env_fpt, agent_PG_fpt ,test_episodes= 1, \
	folder= "/home/huyle/MyGit/Stock-Trading-Environment/weights",\
	name= "/fpt_agent_PG_1e-05.h5" )
cv2.imwrite("fpt_PG.jpg", img_pg)

img_DQN = agent_DQN_fpt.test(save_folder="/home/huyle/MyGit/Stock-Trading-Environment/weights", \
	filename= "FPT_DQN_Episode(5501).h5", visualize= True)

cv2.imwrite("fpt_DQN.jpg", img_DQN)