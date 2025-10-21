import numpy as np
import matplotlib.pyplot as plt

def methode_rectangle(f, a, b, n, type_rectangle='milieu'):
    """
    Méthode des rectangles pour l'intégration numérique
    
    Args:
        f: fonction à intégrer
        a: borne inférieure
        b: borne supérieure
        n: nombre de sous-intervalles
        type_rectangle: 'gauche', 'droit' ou 'milieu'
    
    Returns:
        intégrale_approx: valeur approchée de l'intégrale
    """
    h = (b - a) / n  # largeur des rectangles
    integrale = 0
    
    for i in range(n):
        if type_rectangle == 'gauche':
            x_i = a + i * h  # point gauche
        elif type_rectangle == 'droit':
            x_i = a + (i + 1) * h  # point droit
        elif type_rectangle == 'milieu':
            x_i = a + (i + 0.5) * h  # point milieu
        
        integrale += f(x_i)
    
    return integrale * h

# Exemple d'utilisation
def f(x):
    return x**4 + np.cos(x)

a, b = 0, 2
n = 100

resultat_gauche = methode_rectangle(f, a, b, n, 'gauche')
resultat_droit = methode_rectangle(f, a, b, n, 'droit')
resultat_milieu = methode_rectangle(f, a, b, n, 'milieu')

print("Méthode des Rectangles:")
print(f"Rectangle gauche: {resultat_gauche:.6f}")
print(f"Rectangle droit: {resultat_droit:.6f}")
print(f"Rectangle milieu: {resultat_milieu:.6f}")

#--------------------------------------------------------------

# methode trapeze
def methode_trapeze(f, a, b, n):
    """
    Méthode des trapèzes pour l'intégration numérique
    
    Args:
        f: fonction à intégrer
        a: borne inférieure
        b: borne supérieure
        n: nombre de sous-intervalles
    
    Returns:
        integrale_approx: valeur approchée de l'intégrale
    """
    h = (b - a) / n
    integrale = (f(a) + f(b)) / 2  # termes aux bords
    
    # Somme des points intérieurs
    for i in range(1, n):
        x_i = a + i * h
        integrale += f(x_i)
    
    return integrale * h

def methode_trapeze_vectorise(f, a, b, n):
    """
    Version vectorisée plus efficace
    """
    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n
    
    integrale = h * (0.5 * y[0] + 0.5 * y[-1] + np.sum(y[1:-1]))
    return integrale

# Test
resultat_trapeze = methode_trapeze(f, a, b, n)
resultat_trapeze_vec = methode_trapeze_vectorise(f, a, b, n)

print(f"\nMéthode des Trapèzes: {resultat_trapeze:.6f}")
print(f"Trapèzes: {resultat_trapeze_vec:.6f}")


#--------------------------------------------------------------

# methode de simpson

def methode_simpson(f, a, b, n):
    """
    Méthode de Simpson pour l'intégration numérique
    Note: n doit être pair pour Simpson
    """
    if n % 2 != 0:
        n += 1  # On s'assure que n est pair
        print(f"n ajusté à {n} (doit être pair pour Simpson)")
    
    h = (b - a) / n
    integrale = f(a) + f(b)
    
    # Somme des termes avec coefficients 4 et 2
    for i in range(1, n):
        x_i = a + i * h
        if i % 2 == 1:  # indices impairs
            integrale += 4 * f(x_i)
        else:  # indices pairs
            integrale += 2 * f(x_i)
    
    return integrale * h / 3

def methode_simpson_vectorise(f, a, b, n):
    """
    Version vectorisée de la méthode de Simpson
    """
    if n % 2 != 0:
        n += 1
    
    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n
    
    # Coefficients: 1, 4, 2, 4, 2, ..., 4, 1
    coefficients = np.ones(n + 1)
    coefficients[1:n:2] = 4  # indices impairs
    coefficients[2:n-1:2] = 2  # indices pairs
    
    integrale = h / 3 * np.sum(coefficients * y)
    return integrale

# Test
resultat_simpson = methode_simpson(f, a, b, n)
resultat_simpson_vec = methode_simpson_vectorise(f, a, b, n)

print(f"\nMéthode de Simpson: {resultat_simpson:.6f}")
print(f"Simpson : {resultat_simpson_vec:.6f}")