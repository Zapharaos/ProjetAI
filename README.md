# ProjetAI

## Installation

Si vous n'avez pas encore d'environnement virtuel, créez en un avec virtualenv.

(Seulement dans le cas où vous ne souhaitez pas installer vos dépendances sur votre système)


Lancer l'environnement virtuel:

Sur Linux: `source venv/Scripts/activate`

Sur Windows: `call venv\Scripts\activate.bat`

Installer les dépendances pip :

`pip install -r requirements.txt`

## Utilisation

Il suffit d'executer le script main.py pour afficher l'aide, comme suivant:

`python main.py -h`

Pour l'argument de position, il y'a 3 possibilités:

- 0 : Calculer un réseau de neurones (en utilisant les arguments optionnels)
- 1 : Executer les calculs pour la partie 2 (seulement réseaux neurones)
- 2 : Executer les calculs pour la partie 3 (confusion et tableaux)