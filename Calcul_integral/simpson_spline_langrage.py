import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange

def methode_simpson(f, a, b, n):
    """Méthode de Simpson classique pour l'intégration"""
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return (h / 3) * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]) + y[-1])

def valeur_analytique(a, b):
    """Fonction pour calculer la valeur analytique de l'intégrale"""
    return -np.cos(b) + np.cos(a) + (b**3)/3 - (a**3)/3 + (b**2)/2 - (a**2)/2 - 0.5*(b - a)

def simpson_lagrange(f, a, b, n):
    if n % 2 != 0:
        n += 1
    
    h = (b - a) / n
    x_points = np.linspace(a, b, n + 1)
    integrale_totale = 0
    
    for i in range(0, n, 2):
        x_triplet = x_points[i:i+3]
        y_triplet = f(x_triplet)
        
        poly = lagrange(x_triplet, y_triplet)
        
        x0, x2 = x_triplet[0], x_triplet[2]
        coeffs = poly.coeffs
        
        if len(coeffs) == 3:
            A, B, C = coeffs
        else:
            C, B, A = coeffs
        
        integrale_segment = (A/3 * (x2**3 - x0**3) + 
                             B/2 * (x2**2 - x0**2) + 
                             C * (x2 - x0))
        
        integrale_totale += integrale_segment
    
    return integrale_totale

def demonstration_lagrange_detaille(f, a, b, n_segments=2):
    print("=" * 60)
    print("DÉMONSTRATION DÉTAILLÉE LAGRANGE")
    print("=" * 60)
    
    n_points = n_segments * 2 + 1
    x = np.linspace(a, b, n_points)
    y = f(x)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    x_fine = np.linspace(a, b, 1000)
    y_fine = f(x_fine)
    axes[0, 0].plot(x_fine, y_fine, 'b-', linewidth=2, label='f(x) originale')
    axes[0, 0].scatter(x, y, color='red', s=50, zorder=5, label='Points d\'interpolation')
    axes[0, 0].set_title('Fonction originale et points d\'échantillonnage')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    for segment in range(n_segments):
        i = segment * 2
        x_triplet = x[i:i+3]
        y_triplet = y[i:i+3]
        
        poly = lagrange(x_triplet, y_triplet)
        
        print(f"\nSegment {segment + 1}: Points {x_triplet}")
        print(f"Valeurs: {y_triplet}")
        print(f"Polynôme de Lagrange: {poly}")
        
        x_segment_fine = np.linspace(x_triplet[0], x_triplet[2], 100)
        y_poly = poly(x_segment_fine)
        y_exact = f(x_segment_fine)
        
        axes[0, 1].plot(x_segment_fine, y_exact, 'b-', alpha=0.5, label='Fonction exacte' if segment == 0 else "")
        axes[0, 1].plot(x_segment_fine, y_poly, '--', linewidth=2, label=f'Polynôme segment {segment+1}')
        axes[0, 1].scatter(x_triplet, y_triplet, color='red', s=50, zorder=5)
    
    axes[0, 1].set_title('Interpolation de Lagrange par segments')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    integrale_simpson = simpson_lagrange(f, a, b, n_segments * 2)
    integrale_classique = methode_simpson(f, a, b, n_segments * 2)
    
    print(f"\nRÉSULTATS D'INTÉGRATION:")
    print(f"Méthode Simpson-Lagrange: {integrale_simpson:.8f}")
    print(f"Méthode Simpson classique: {integrale_classique:.8f}")
    print(f"Différence: {abs(integrale_simpson - integrale_classique):.2e}")
    
    n_values = [4, 8, 16, 32, 64]
    erreurs_lagrange = []
    erreurs_classique = []
    
    for n in n_values:
        exact = valeur_analytique(a, b)
        lagr = simpson_lagrange(f, a, b, n)
        clas = methode_simpson(f, a, b, n)
        erreurs_lagrange.append(abs(lagr - exact))
        erreurs_classique.append(abs(clas - exact))
    
    axes[1, 0].loglog(n_values, erreurs_lagrange, 'o-', label='Simpson-Lagrange')
    axes[1, 0].loglog(n_values, erreurs_classique, 's-', label='Simpson classique')
    axes[1, 0].set_xlabel('Nombre de points')
    axes[1, 0].set_ylabel('Erreur')
    axes[1, 0].set_title('Convergence des méthodes')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return integrale_simpson

# Exemple d'utilisation
def f(x):
    return np.sin(x) + x**2 - 0.5*x + 1
