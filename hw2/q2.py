import numpy as np
import matplotlib.pyplot as plt



def f0(x):
    return np.sin(x[0] + x[1]) + (x[0] - x[1])**2 - 1.5*x[0] + 2.5*x[1] + 1

def grad_f0(x):
    return np.array([np.cos(x[0] + x[1]) + 2*(x[0] - x[1]) - 1.5, np.cos(x[0] + x[1]) - 2*(x[0] - x[1]) + 2.5])

def gradient_method(x0, L, epsilon):
    x = x0
    ite = 0
    norm_grad_f = np.zeros(60) #pour les plots
    while np.linalg.norm(grad_f0(x)) > epsilon:
        norm_grad_f[ite] = np.linalg.norm(grad_f0(x))
        x = x - 1/L * grad_f0(x)
        ite+=1
    return x, ite, norm_grad_f

def derivative_free_method(x0, L, epsilon):
    x = x0
    n = len(x0)
    h = epsilon / (4 * L * np.sqrt(n))
    ite = 0
    e = np.eye(n)
    #Définissons 2 vecteurs qu'on utilisera pour les plots
    norm_g = np.zeros(10001)
    expression = np.zeros(10001)
    while np.linalg.norm(grad_f0(x)) > epsilon and ite < 10000:
        g = np.array([(f0(x + h * e[i]) - f0(x)) / h for i in range(n)])
        expression[ite] = (f0(x + h * e[0]) - f0(x))
        x = x - (1 / L) * g
        norm = np.linalg.norm(g)
        norm_g[ite] = norm
        ite+=1
    if ite == 10000:
        ite = "F"
    return x, ite, norm_g,expression

x0 = np.array([-3, 4])
L = 4
epsilons = [10**(-i) for i in range(3, 14, 2)]
norm_g_tot = []
expression_tot = []

print("epsilon\tGradient method\tDerivative-free method")
for i in range(len(epsilons)):
    x, ite, norm_g = gradient_method(x0, L, epsilons[i])
    gm_iters = str(ite)
    print(f"epsilon: {epsilons[i]} + iterations grad : {ite}")
    #print(f"x = {x}")
    norm_g_tot.append(norm_g)
    x, ite, norm_g, expression = derivative_free_method(x0, L, epsilons[i])
    dfm_iters = str(ite)
    expression_tot.append(expression)
    print(f"epsilon: {epsilons[i]} + iterations deriv free : {ite}")
    #print(f"x = {x}")
    norm_g_tot.append(norm_g)
    n = len(norm_g_tot[-1])





plt.figure()
plt.title("Cause of the problems")
plt.xlabel("iterations")
plt.ylabel("f(x+hei-f(x))")
plt.yscale("log")
for i in range(len(expression_tot)):
    plt.plot(range(60), expression_tot[i][:60], label=f"epsilon={epsilons[i]}")
    plt.legend()
plt.show()

# Define the number of subplots and their arrangement
num_subplots = 6
num_plots_per_subplot = 2
fig, axes = plt.subplots(num_subplots, 1, figsize=(8, 12))

# Create a list of epsilon values for labeling
epsilons = [10**(-i) for i in range(3, 14, 2)]

# Iterate over the subplots
for i in range(num_subplots):
    ax = axes[i]
    ax.set_title(f"Epsilon = {epsilons[i]}", y=1.3)
    ax.set_xlabel("iterations")
    ax.set_ylabel("norm")
    ax.set_yscale("log")

    # Iterate over the plots in each subplot
    for j in range(num_plots_per_subplot):
        index = i * num_plots_per_subplot + j
        if index < len(norm_g_tot):
            if index % 2 == 0:
                ax.plot(range(len(norm_g_tot[0])), norm_g_tot[index][:60], label="Gradient method")
                ax.legend()
            else :
                ax.plot(range(len(norm_g_tot[0])), norm_g_tot[index][:60], label="Deriv Free method")
                ax.legend()

plt.tight_layout()  # Ensure proper spacing between subplots
plt.show()