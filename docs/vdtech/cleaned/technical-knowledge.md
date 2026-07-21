# VDtec — Connaissances Techniques

## 1. Tarifs d'assistance

| Type d'intervention | Tarif | Notes |
|---|---|---|
| **Sur site (On-site)** | 2 200 Rs + TVA | Tarif minimal, sujet à révision annuelle |
| **À distance (Remote)** | 1 100 Rs + TVA | Déclenché automatiquement après approbation client |

---

## 2. Architecture : Machine Standard vs Système avec Contrôleur

### Option 1 — Machine autonome (standard)
- La machine décide directement de l'ouverture de la porte en combinant ses propres données et le logiciel.
- **Avantage :** Plus simple à gérer au quotidien.
- **Limite sécurité :** Si la machine est arrachée du mur et les câbles court-circuités (câbles de commande de la gâche reliés directement à l'appareil extérieur), la porte peut être ouverte en moins de 2 minutes.

### Option 2 — Système avec Contrôleur (recommandé pour la haute sécurité)
- Le contrôleur centralise l'intelligence du système et valide ou refuse les accès.
- Le contrôleur est installé **à l'intérieur des locaux**, inaccessible de l'extérieur.
- L'alimentation de la serrure/gâche ou de l'aimant est reliée directement au contrôleur.
- Si le lien entre le lecteur extérieur et le contrôleur est coupé, le contrôleur détecte le sabotage (*tampering*) et **maintient la porte verrouillée**.
- Permet la gestion fine des droits : accès différencié par utilisateur, par porte, par plage horaire.

---

## 3. Comparatif des Marques de Contrôleurs

| Critère | ZKTeco | Honeywell |
|---|---|---|
| **Positionnement** | Milieu de gamme, fiable | Haut de gamme, très sécurisé |
| **Prix contrôleur complet** | ~20 300 Rs | ~100 000 Rs (matériel seul) |
| **Logiciel** | Inclus dans la gamme | Peut atteindre ~400 000 Rs |
| **Cas d'usage** | PME, entreprises de taille moyenne | Grands projets, infrastructures critiques (ex : aéroports) |
| **Exemple projet** | Mauritius Network Services (extension système existant) | Projets institutionnels ou gouvernementaux |

---

## 4. Accessoires de Sécurité et Normes d'Évacuation

Toute installation de contrôle d'accès doit intégrer des dispositifs de déverrouillage d'urgence :

| Accessoire | Emplacement | Rôle |
|---|---|---|
| **Bouton poussoir (Exit button)** | Intérieur | Sortie courante sans authentification |
| **Boîtier bris de glace (Breakglass) — vert** | Intérieur (obligatoire) | Urgence incendie : casse la vitre pour couper l'alimentation et libérer la porte immédiatement |
| **Interrupteur à clé maîtresse (Key switch)** | Extérieur | Déverrouillage manuel par administrateur en cas de panne du système |
| **Ferme-porte (Door closer)** | Sur la porte | Referme automatiquement la porte |

**Systèmes de verrouillage :**
- **Gâches électroniques** ou **ventouses magnétiques (aimants)**
- Les aimants affichent un voyant **vert** (verrouillé) ou **rouge** (défaut d'alignement ou porte ouverte)

---

## 5. Alimentation et Autonomie

| Paramètre | Valeur |
|---|---|
| **Autonomie batterie standard** | 4 à 6 heures |
| **Autonomie maximale** | 10 à 12 heures (selon modèle et fréquence d'utilisation) |
| **Comportement en cas de décharge complète (aimant)** | La porte s'ouvre automatiquement par sécurité (*fail-safe*) |
| **Remplacement des batteries** | Tous les **2 ans** |
| **Facturation remplacement batterie** | Facturé séparément — non inclus dans le contrat de maintenance standard |

---

## 6. Câblage et Réseau

- **Visite technique préalable obligatoire :** le coût du câblage ne peut jamais être estimé sans analyse des passages de câbles sur site.
- **Réseau fermé (local) :** les systèmes fonctionnent sur un réseau fermé, mais peuvent être interconnectés entre étages ou sites via internet vers un serveur central.
- **Responsabilité client (Time & Attendance standard) :** pour une installation simple sans verrouillage de porte, le client doit fournir le point d'alimentation électrique et la prise réseau RJ45 à l'emplacement de la machine.
- **Type de connexion :** majoritairement filaire (câble réseau RJ45).
- **Cas fibre optique :** un **convertisseur de média (media converter)** est requis.
- **Wi-Fi :** certains modules proposent une option Wi-Fi selon le modèle.
