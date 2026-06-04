## 1. Tarifs et types d'assistance (VDtec)

Les demandes d'assistance technique de VDtec sont fréquentes et suivent une tarification claire :

- **Assistance sur site (On-site) :** Le tarif minimal actuel est de **2 200 Rs + TVA**. Ce montant est sujet à évolution après un an, mais constitue la base actuelle.
- **Assistance à distance (Remote) :** Ce type d'intervention est facturé par défaut à **1 100 Rs + TVA**. Ce processus est automatisé : dès qu'un client demande une vérification à distance, ce tarif standard s'applique après son approbation.

---

## 2. Choix du matériel : Architecture Standard vs Contrôleur

Arshad distingue deux niveaux de sécurité pour les systèmes de contrôle d'accès :

### Option 1 : La machine standard (autonome)

- **Fonctionnement :** La machine décide directement de l'ouverture en combinant ses propres données avec le logiciel.
- **Avantages :** Système plus simple et plus facile à gérer au quotidien.
- **Inconvénients (Sécurité faible) :** Si un intrus arrache la machine du mur et court-circuite les fils (les câbles de commande de la gâche étant directement reliés à l'appareil extérieur), il peut déverrouiller la porte en moins de deux minutes.

### Option 2 : Le système avec Contrôleur (Recommandé pour la haute sécurité)

- **Fonctionnement :** Le contrôleur centralise « l'intelligence » du système. C'est lui qui valide ou non l'accès.
- **Mécanisme de sécurité :** Même si la machine extérieure (le lecteur) est arrachée ou manipulée, la porte reste fermée. L'alimentation de la serrure/gâche électrique ou de l'aimant est reliée directement au contrôleur, qui est dissimulé à l'intérieur des locaux et donc inaccessible. Si le lien entre le lecteur et le contrôleur est coupé, le contrôleur détecte un sabotage (*tampering*) et maintient la porte verrouillée.
- **Gestion des droits :** C'est le contrôleur qui vérifie si un utilisateur enregistré a le droit spécifique d'ouvrir une porte précise à une heure donnée (ex: accès différencié entre Samuel et Arshad selon les portes).

---

## 3. Les Marques de Contrôleurs : ZKTECO vs HONEYWELL

Le choix du contrôleur dépend de la taille et des exigences de sécurité du projet :

- **ZKTECO :** Solution fiable, moins excessive en termes de prix et de maintenance. Adaptée aux entreprises de taille moyenne (ex: projet en cours pour *Mauritius Network Services* qui étend son système existant).
- **HONEYWELL :** Matériel haut de gamme, beaucoup plus sécurisé, mais nettement plus onéreux. À titre de comparaison, là où un contrôleur ZKTECO complet coûte environ **20 300 Rs**, une configuration équivalente chez Honeywell (nécessitant un *Master Controller* et des *Reader Boards*) fait grimper le budget matériel aux alentours de **100 000 Rs** (hors coût du logiciel qui peut avoisiner les 400 000 Rs). Honeywell est réservé aux gros projets ou infrastructures critiques (ex: aéroports).

---

## 4. Accessoires de Sécurité et Normes d'Évacuation

Toute installation de contrôle d'accès doit intégrer des dispositifs de déverrouillage d'urgence légaux :

- **Bouton poussoir (Exit button) :** Installé à l'intérieur pour la sortie courante.
- **Boîtier de bris de glace vert (Breakglass) :** Obligatoire à **l'intérieur** des bureaux. En cas d'urgence (incendie), il permet à n'importe quel employé de briser la vitre pour couper l'alimentation et libérer immédiatement la porte sans passer par le système électronique.
- **Interrupteur à clé maîtresse (Key switch) :** Installé à **l'extérieur**. Il permet à un administrateur d'utiliser une clé physique pour déverrouiller manuellement la porte si le système est en panne ou si les employés sont bloqués dehors.
- **Systèmes de verrouillage :** Choix entre des gâches électroniques ou des ventouses magnétiques (aimants). Les aimants affichent un voyant vert lorsqu'ils sont verrouillés et un voyant rouge en cas de défaut d'alignement ou d'ouverture.
- **Ferme-porte (Door closer)**.

---

## 5. Gestion de l'Alimentation et Autonomie (Secours)

- **Batterie de secours :** Les systèmes intègrent un module d'alimentation avec une batterie interne pour pallier les coupures d'électricité.
- **Autonomie :** La batterie offre une autonomie moyenne de **4 à 6 heures**, mais cela peut atteindre 10 à 12 heures selon le modèle d'appareil et la fréquence d'utilisation de la porte. En cas de décharge complète d'un système sur aimant (ventouse), la porte s'ouvre automatiquement par sécurité.
- **Maintenance :** Les batteries doivent être remplacées **tous les 2 ans**. Ce remplacement de pièce est facturé séparément et n'est pas inclus d'office dans le coût du contrat de maintenance standard (qui ne couvre que le support technique).

---

## 6. Câblage et Spécificités Réseau

- **Étude sur site obligatoire :** Le coût du câblage ne peut jamais être estimé sans une visite technique préalable pour analyser les passages de câbles. 
- **Réseau fermé :** Le terme "en local" signifie que le système fonctionne sur un réseau fermé, mais interconnecté. Il est possible de relier des machines situées à des étages différents ou sur d'autres sites via internet pour les faire communiquer avec le serveur central.
- **Responsabilité du client (Time & Attendance standard) :** Pour une installation simple de gestion des présences (sans verrouillage de porte), le client doit fournir lui-même le point d'alimentation électrique et la prise réseau (RJ45) à l'emplacement de la machine.
- **Type de connexion :** Les appareils VDtec utilisent majoritairement des connexions filaires (câble réseau RJ45). Si le client utilise de la fibre optique, un convertisseur de média (*media converter*) est requis. Certains modules spécifiques proposent également une option Wi-Fi selon les modèles.

