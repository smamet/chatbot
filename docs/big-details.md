Voici un document de référence complet, structuré et optimisé pour être intégré dans une **base de données vectorielle (RAG)**.

Pour faciliter le travail de classification et de réponse de votre agent IA, le document est divisé en sections sémantiques claires avec des mots-clés spécifiques. Cela permettra à l'agent de bien différencier une requête **Commerciale (Vente)** d'une requête **Technico-Commerciale / SAV (Après-vente)**.

---

# DOCUMENT DE RÉFÉRENCE CONNAISSANCE : VDTEC & EASYWEB

## 1. IDENTITÉ ET PRÉSENTATION DE L'ENTREPRISE

* **Nom de la compagnie :** VDtec
* **Partenaire stratégique / Entreprise sœur :** Easyweb
* **Secteur d'activité :** B2B – Solutions de Sécurité, Gestion du Temps (Pointage), Matériel Informatique, Infrastructure Réseau et Logiciels RH.
* **Public cible :** Entreprises, PME, Grandes Entreprises, Responsables RH, Directeurs Financiers, Responsables Sécurité/IT.
* **Philosophie commerciale :** Approche consultative et personnalisée. Aucune vente n'est réalisée sans une présentation ou une étude technique préalable.

---

## 2. MODULE COMMERCIAL : SERVICES DE VENTE (AVANT-VENTE)

*Cette section sert à alimenter l'agent IA pour la qualification des leads, l'envoi de fiches techniques et l'explication du catalogue.*

### A. Catalogue Produits et Solutions (Mots-clés : Acheter, Prix, Devis, Matériel, Catalogue, Installer)

1. **Systèmes de Pointage et de Gestion des Présences (Time & Attendance) :**
* Terminaux physiques de pointage (Reconnaissance faciale, Empreinte digitale, RFID/Cartes).
* *Note technique importante :* Pour une installation standard (Time & Attendance seul sans contrôle de porte), le client doit obligatoirement fournir la prise réseau RJ45 et le point d'alimentation électrique à l'emplacement souhaité de la machine.


2. **Systèmes de Contrôle d'Accès et Haute Sécurité :**
* **Architecture Standard (Autonome) :** La machine gère l'ouverture. Plus simple, mais sécurité modérée (les câbles de commande de gâche sont à l'extérieur).
* **Architecture Haute Sécurité avec Contrôleur :** L'intelligence est déportée dans un contrôleur dissimulé à l'intérieur des locaux. En cas d'arrachage du lecteur extérieur, la porte reste verrouillée. Permet de gérer des droits d'accès fins par utilisateur/porte/horaire.
* *Marques proposées :* **ZKTECO** (Excellent rapport qualité/prix, idéal PME, ex: ~20 300 Rs pour un contrôleur complet) ; **HONEYWELL** (Très haut de gamme, infrastructures critiques, matériel ~100 000 Rs, logiciel ~400 000 Rs).
* *Accessoires obligatoires pour conformité légale/évacuation :* Bouton poussoir (sortie normale), Boîtier bris de glace vert (*Breakglass* intérieur pour urgence/incendie), Interrupteur à clé maîtresse (*Key switch* extérieur pour ouverture manuelle en cas de panne globale), Ferme-porte, Ventouses magnétiques (aimants avec voyants LED vert/rouge) ou gâches électriques.


3. **Gestion des Rondes (Guard Patrol) :**
* Équipements de traçabilité pour agents de sécurité. Permet de vérifier les points de passage (Points A, B, C) et de valider les rondes.


4. **Matériel Informatique & Licences :**
* Vente d'ordinateurs portables professionnels (Partenaire de la marque HP).
* Vente de licences Microsoft (Office 365, Azure, Services Cloud).


5. **Systèmes de Gestion des Actifs (Asset Management) :**
* Solutions de traçabilité de matériel et d'inventaire via codes-barres ou technologies RFID.


6. **Imprimantes à Cartes et Consommables :**
* Systèmes d'impression de badges d'identification (Partenaire **EDIKIO** / **EVOLIS**).
* Gamme disponible : *Edikio Price Tag* (Access, Flex, Duplex pour étiquettes de prix), *Edikio Guest* (Access, Flex pour l'hôtellerie/accueil), imprimantes *Badgy 200, Zenius 1 & 2, Primacy*.
* Consommables : Rubans monochromes et couleur (BlackFlex, Black RMS), kits de nettoyage standards et avancés, cartes vierges en PVC (conformes aux normes de sécurité alimentaire pour la gamme Edikio).



### B. Synergie logicielle avec Easyweb (Mots-clés : Logiciel, RH, Paie, Payroll, Application, eTime, HRMS)

VDtec travaille en parfaite intégration avec le logiciel **HRMS d’Easyweb** (proposé sous forme d'abonnement mensuel). Le logiciel comprend 4 modules :

1. **HR (Ressources Humaines) :** Gestion des employés, contrats, sanctions, avantages.
2. **Self Service :** Application/Portail employé pour télécharger les fiches de paie, déclarer les heures sup et pointer à distance via géolocalisation GPS.
3. **Payroll :** Calcul automatique de la paie.
4. **Time & Attendance :** Récupération automatique des données de pointage des machines VDtec.

* **Le connecteur eTime :** VDtec extrait les données brutes (*raw data*) de ses machines et utilise le logiciel **eTime** (fourni par Easyweb) pour traiter les heures supplémentaires et les absences.
* **Flexibilité (Logiciels Tiers) :** Si le client utilise déjà un autre ERP/logiciel de paie (ex: Sicorax), VDtec peut développer des API pour y interconnecter ses terminaux de pointage. Cette liaison API fait l'objet d'une facturation sous forme de **redevance mensuelle**.

### C. Processus de Vente Strict (Mots-clés : Intéressé, Commander, Acheter, Rendez-vous)

L'agent commercial IA doit guider le prospect à travers ces étapes incontournables :

1. **Prise de RDV / Présentation Obligatoire :** Aucune commande n'est validée sans une rencontre et une étude des besoins. L'IA peut envoyer des fiches techniques avec des prix indicatifs, mais doit inciter au RDV.
2. **Envoi de la proposition commerciale** après étude.
3. **Signature du Bon de Commande.**
4. **Achat du matériel** (si hors stock) & Planification de l'installation.
5. **Installation physique, câblage, enregistrement des utilisateurs et Formation** (prestation payante).
6. **Période de Fine-Tuning (1 mois gratuit) :** Le premier mois suivant l'installation, tous les ajustements logiciels et techniques sont entièrement gratuits.
7. **Proposition du contrat de maintenance.**

---

## 3. MODULE TECHNICO-COMMERCIAL : SAV, MAINTENANCE ET APRÈS-VENTE

*Cette section sert à alimenter l'agent IA pour répondre aux clients existants ayant un problème, une panne ou une demande de modification.*

### A. Grille Tarifaire Hors Contrat (Mots-clés : Panne, Bug, Assistance, Ticket, Facture, Problème, Technicien)

Si un client demande une intervention et qu'il n'a pas de contrat actif, ou que l'élément n'est pas couvert, les tarifs réglementaires s'appliquent après accord du client :

* **Assistance technique à distance (Télé-assistance / Remote) :** Facturation forfaitaire automatisée de **1 100 Rs + TVA** par ticket.
* **Intervention technique physique (On-site) :** Facturation minimale de **2 200 Rs + TVA** (sujet à révision après un an).

### B. Le Contrat de Maintenance Annuel - AMC (Mots-clés : Contrat, AMC, Couverture, Renouvellement)

* **Coût :** Généralement calculé à hauteur de **15% de la valeur totale du projet initial** (ajustable selon les besoins spécifiques). Payé mensuellement mais engagement annuel.
* **Avantage Majeur (Le prêt de matériel) :** En cas de panne d'un équipement sous contrat AMC, VDtec s'engage à fournir un **appareil de remplacement temporaire (appareil de prêt)** le temps de réparer ou remplacer la machine défectueuse. Le client ne subit aucune interruption de service.

### C. Règles de Garantie Constructeur vs AMC (Mots-clés : Garantie, Batterie, Cassé, Remplacement)

* La garantie constructeur standard dure généralement 1 à 2 ans selon le matériel.
* **Attention :** La garantie couvre uniquement les pièces défectueuses. Contrairement au contrat AMC, **aucun appareil de prêt n'est fourni** pendant la période de réparation usine si le client n'a pas souscrit à l'AMC.
* **Exclusions de contrat :** Certaines pièces ou consommables n'entrent pas dans le cadre des garanties ou des contrats de maintenance classique (ex: les cartes imprimées, les cartes RFID, l'usure normale).
* **Le cas des batteries de secours :** Les systèmes de sécurité possèdent des batteries assurant **4 à 6 heures d'autonomie** (jusqu'à 12h selon l'usage). Si l'électricité est coupée et la batterie vide, un système sur aimant s'ouvre automatiquement par sécurité. **Ces batteries doivent être remplacées tous les 2 ans.** Ce remplacement de pièce est facturé en supplément et n'est pas inclus dans le coût d'assistance de l'AMC.

### D. Contraintes de Câblage et Déploiement (Mots-clés : Câble, Déplacer, Étage, On-Premise, Serveur, Cloud)

* **Étude sur site obligatoire pour le câblage :** Le coût du câblage dépend de la configuration des lieux (ex: simulation moyenne de ~2 500 Rs). On ne peut jamais donner de prix ferme pour un câblage réseau ou électrique sans visite technique.
* **Architecture Réseau :** Les terminaux communiquent en réseau filaire (RJ45). Si le client utilise de la fibre optique, un convertisseur de média (*media converter*) est obligatoire. Certains modules supportent le Wi-Fi.
* **Multi-sites / Étages :** Le système peut fonctionner en "local interconnecté". Des machines situées sur des sites géographiques différents ou des étages distincts peuvent être reliées au serveur central via des tunnels Internet.
* **Cloud vs On-Premise (Serveur Local) :** Les solutions logicielles s'installent soit sur le Cloud, soit sur les serveurs locaux du client (*On-Premise*). Les installations *On-Premise* sont complexes et requièrent l'ingénierie d'un technicien expert (Arshad) pour s'adapter à l'infrastructure informatique interne du client. Les devis *On-Premise* de base sont donc toujours sujets à des ajustements financiers après l'audit technique.

---

## 4. DIRECTIVES DE COMPORTEMENT POUR L'AGENT IA (PROMPT ENHANCEMENT)

1. **Phase de Classification (Étape 1) :** Dès la réception d'un e-mail, l'agent IA doit analyser si la demande contient des mots-clés du **Module 2 (Commercial)** ou du **Module 3 (Après-Vente)**.
2. **Ton et Posture :** Professionnel, rigoureux, orienté B2B. L'agent ne doit jamais s'engager sur un prix de câblage final ou une installation *On-Premise* sans stipuler "sous réserve d'un audit technique sur site".
3. **Règle d'or Commerciale :** Si un client souhaite passer commande, ne jamais dire "votre commande est validée". Toujours proposer un créneau de rendez-vous pour une présentation ou une étude des besoins en premier lieu.
4. **Règle d'or SAV / Technique :** Si un client hors contrat demande une intervention (à distance ou sur site), l'IA doit explicitement mentionner les tarifs (1 100 Rs pour le remote / 2 200 Rs pour le site + TVA) et demander sa validation avant de transférer le ticket au planning des techniciens. Si le client se plaint d'un délai de réparation sous garantie, vérifier s'il a un contrat AMC : si non, lui rappeler poliment que le prêt de matériel est réservé aux abonnés AMC.