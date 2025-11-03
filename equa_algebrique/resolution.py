import math
import numpy as np
import matplotlib.pyplot as plt

def methode_newton(f, df, x0, epsilon=1e-6, NMAX=100):
   
    
    # f : fonction f(x) dont on cherche la racine
    # df : La dérivée de la fonction f(x)
    # x0 : Point de départ de l'itération
    # epsilon : Précision souhaitée (critère d'arrêt)
    # NMAX : Nombre maximum d'itérations
    # float or None - La racine approximative ou None si convergence échoue
    
    
    n = 0  # Initialisation du compteur d'itérations
    
    while True:
        n += 1  # Incrémentation du compteur
        
        # Sauvegarde de l'ancienne valeur.
        # x va stocker l'ancienne valeur valeur de la condition d'arrêt
        x = x0
        
        # Application de la formule de Newton : x_nouveau = x_ancien - f(x)/f'(x)
        x0 = x - f(x) / df(x)

        # Critères d'arrêt
        # 1. La différence entre deux itérations est suffisamment petite
        # 2. On a atteint le nombre maximum d'itérations
        if abs(x0 - x) <= epsilon or n >= NMAX:
            break
    
    # Vérification de la convergence
    if n == NMAX:
        print("Trop d'itérations - Convergence non atteinte")
        return None
    else:
        print(f"Convergence atteinte en {n} itérations")
        return x0

# utilisation
if __name__ == "__main__":
    # Exemple 1: Trouver la racine de f(x) = x³ - x (racine de 0)
    def f1(x):
        return x**3 - x
    
    def f1_prime(x):
        return 3*x**2 - 1
    
    # Test avec différents points de départ
    racine1 = methode_newton(f1, f1_prime, x0=1.5)
    print(f"Racine approximative de x³ - x: {racine1}")
    print()
    print(f"Vérification: {racine1}³ - {racine1} = {f1(racine1)}")
    print()
    
    # Exemple 2: Trouver la racine de f(x) = cos(x) - x
    def f2(x):
        return math.cos(x) - x
    
    def f2_prime(x):
        return -math.sin(x) - 1
    
    racine2 = methode_newton(f2, f2_prime, x0=0.5)
    print(f"Racine approximative de cos(x) - x: {racine2}")
    print()
    print(f"Vérification: cos({racine2}) - {racine2} = {f2(racine2)}")