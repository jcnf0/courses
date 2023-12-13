import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('Results/out1.csv', skip_blank_lines=False)
print(df)
# Identify the start of each run
run_starts = [0] + df[df['Episode'].isnull()].index.tolist() + [len(df)]
print(run_starts[4])
alpha = [0.95, 0.8, 0.7, 0.6, 0.5]
for k in range(5):
    # Get the reward data for this run
    run_ep = df['Episode'].iloc[run_starts[k]:run_starts[k+1]-1]
    # Get the reward data for this run
    run_rewards = df['Rewards'].iloc[run_starts[k]:run_starts[k+1]-1]
    # Plot the reward data for this run
    plt.plot(run_ep, run_rewards, label='g = {}'.format(alpha[k]))

plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
plt.title('Reward per Episode for Last 5 A2C Runs')
plt.savefig('Results/gamma.pdf')
plt.show()
