import csv
import matplotlib.pyplot as plt

def load_log_data(filename):
    episodes = []
    total_rewards = []
    steps = []
    epsilons = []
    
    try:
        with open(filename, "r") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                episodes.append(int(row["Episode"]))
                total_rewards.append(float(row["Total Reward"]))
                steps.append(int(row["Steps"]))
                epsilons.append(float(row["Epsilon"]))
    except FileNotFoundError:
        print(f"File {filename} not found.")
    return episodes, total_rewards, steps, epsilons

def plot_progression(episodes, total_rewards, steps, epsilons):
    plt.figure(figsize=(12, 10))
    
    # Plot total rewards over episodes.
    plt.subplot(3, 1, 1)
    plt.plot(episodes, total_rewards, marker='o', linestyle='-', color='b')
    plt.title("Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    
    # Plot steps over episodes.
    plt.subplot(3, 1, 2)
    plt.plot(episodes, steps, marker='o', linestyle='-', color='g')
    plt.title("Steps per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.grid(True)
    
    # Plot epsilon value over episodes.
    plt.subplot(3, 1, 3)
    plt.plot(episodes, epsilons, marker='o', linestyle='-', color='r')
    plt.title("Epsilon per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    log_filename = "dodge_ai_log.csv"
    episodes, total_rewards, steps, epsilons = load_log_data(log_filename)
    
    if episodes:
        plot_progression(episodes, total_rewards, steps, epsilons)
    else:
        print("No data to plot. Please ensure the log file exists and has data.")

