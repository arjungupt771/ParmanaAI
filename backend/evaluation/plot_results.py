import matplotlib.pyplot as plt

def plot_distribution(scores_parmana, scores_baseline):
    plt.hist(scores_parmana, bins=10, alpha=0.6, label="Parmana AI")
    plt.hist(scores_baseline, bins=10, alpha=0.6, label="Vanilla LLM")

    plt.xlabel("Hallucination Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("Hallucination Score Distribution")
    plt.show()