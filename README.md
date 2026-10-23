# Imputation et Reconstruction de Courbes de Consommation Électrique par Apprentissage Profond

Projet développé dans le cadre du Hackathon Enedis (Décembre 2025).

## Présentation du Projet

L'imputation de données manquantes dans les séries temporelles de consommation électrique est un enjeu stratégique pour la gestion et le dimensionnement des réseaux de distribution d'électricité. L'objectif de ce projet est d'imputer efficacement les manques (« trous ») présents dans des milliers de courbes de charge temporelles.

### Approche Méthodologique :
Plutôt que d'utiliser une simple interpolation linéaire, le modèle combine une technique de **Clustering K-Means** et un **Réseau de Neurones Perceptron Multi-Couches (MLP / Keras)** pour prédire et corriger le résidu de consommation :

1. **Prétraitement & Interpolation** : Premier lissage des séries temporelles trouées par interpolation linéaire.
2. **Double Clustering K-Means (`scikit-learn`)** :
   * **Profils clients** : Partitionnement en 4 clusters pour identifier la typologie de consommation des courbes.
   * **Saisonnalité temporelle** : Partitionnement en 3 clusters sur les horodates pour capturer les variations journalières.
3. **Apprentissage des Résidus (Perceptron / Keras)** :
   * Entraînement d'un réseau de neurones séquentiel (128 $\to$ 64 $\to$ 1 neurone) prenant en entrée l'interpolation initiale enrichie des encodages de clusters.
   * Le réseau apprend à prédire la déviation (résidu) par rapport à l'interpolation linéaire simple pour reconstituer le signal réel.

---

## Stack Technique & Dépendances

* **Manipulation de données** : `pandas`, `numpy`
* **Machine Learning & Preprocessing** : `scikit-learn` (`KMeans`, `StandardScaler`, `mean_absolute_error`, `r2_score`)
* **Deep Learning** : `tensorflow`, `keras`
* **Visualisation** : `matplotlib`, `seaborn`

---

## 📊 Performances du Modèle

Sur l'ensemble de test comportant des courbes de charge fortement dégradées :
* **MAE (Erreur Absolue Moyenne)** : $\approx 103.1$ kW
* **Score $R^2$** : $\mathbf{0.768}$ (Le modèle explique près de $77\%$ de la variance du signal manquant)
