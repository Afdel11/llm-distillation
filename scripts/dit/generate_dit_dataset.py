#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Génère un jeu d'instructions/réponses sur le Dakar Institute of Technology
(DIT), à partir de faits vérifiés sur dit.sn et fr.wikipedia.org (consultés
le 21 août 2026). Tout le contenu factuel est reformulé avec des mots
originaux -- aucune phrase copiée telle quelle des sources.

Chaque entrée factuelle est déclinée en plusieurs formulations de question,
pour donner au Student plusieurs façons de "voir" la même information --
une stratégie légitime d'augmentation de données sur un domaine étroit,
distincte de l'invention de faits.
"""

import json
import random

random.seed(0)

# =====================================================================
# Base factuelle (topic -> liste de (question, réponse))
# =====================================================================

DATA = []


def add(questions, response):
    """Associe plusieurs formulations de question à UNE réponse vérifiée."""
    for q in questions:
        DATA.append({"prompt": q, "response": response})


# --- Histoire et identité -------------------------------------------
add(
    [
        "Qu'est-ce que le Dakar Institute of Technology ?",
        "Peux-tu me présenter le DIT ?",
        "Qu'est-ce que le DIT ?",
        "En quoi consiste le Dakar Institute of Technology ?",
    ],
    "Le Dakar Institute of Technology (DIT) est une école supérieure privée basée à Dakar, au Sénégal, "
    "spécialisée dans l'informatique, le Big Data et l'intelligence artificielle. C'est la première "
    "école d'Afrique de l'Ouest entièrement dédiée à ces domaines."
)

add(
    [
        "Quand le DIT a-t-il été fondé ?",
        "En quelle année le Dakar Institute of Technology a-t-il été créé ?",
        "Depuis quand existe le DIT ?",
    ],
    "Le Dakar Institute of Technology a été fondé en 2019. Les premières formations certifiantes ont "
    "démarré cette année-là, avant l'ouverture de formations diplômantes à partir de 2021."
)

add(
    [
        "Qui a fondé le Dakar Institute of Technology ?",
        "Qui est le fondateur du DIT ?",
        "Qui dirige le DIT ?",
        "Qui est le directeur général du Dakar Institute of Technology ?",
    ],
    "Le DIT a été fondé par le Dr Nicolas Poussielgue, qui en est aujourd'hui encore le directeur "
    "général. Il a créé l'école dans le but de contribuer au développement technologique de l'Afrique "
    "en formant localement des professionnels compétents."
)

add(
    [
        "Quelle est la devise du DIT ?",
        "Quel est le slogan du Dakar Institute of Technology ?",
    ],
    "La devise du Dakar Institute of Technology est « Codez votre Futur »."
)

add(
    [
        "Pourquoi le DIT se concentre-t-il sur le Big Data et l'intelligence artificielle ?",
        "Pourquoi avoir choisi le Big Data et l'IA comme spécialités ?",
    ],
    "Le DIT a choisi de se concentrer sur le Big Data et l'intelligence artificielle parce que ce sont "
    "des domaines très demandés sur le marché de l'emploi. L'école veut préparer ses étudiants au "
    "dynamisme de ce marché avec des formations axées sur la pratique."
)

# --- Localisation et contact ------------------------------------------
add(
    [
        "Où se trouve le Dakar Institute of Technology ?",
        "Quelle est l'adresse du DIT ?",
        "Où est situé le DIT à Dakar ?",
    ],
    "Le Dakar Institute of Technology est situé à l'Immeuble 46, Cité Keur Gorgui, à Dakar, au Sénégal — "
    "à proximité immédiate de grands groupes technologiques implantés dans le pays, comme Orange."
)

add(
    [
        "Comment contacter le DIT ?",
        "Quel est le numéro de téléphone du Dakar Institute of Technology ?",
        "Comment joindre le DIT par téléphone ou WhatsApp ?",
        "Quelle est l'adresse e-mail du DIT ?",
    ],
    "Le DIT peut être contacté par WhatsApp au +221 77 308 92 92, par téléphone fixe au "
    "+221 33 822 47 33, ou par e-mail à info@dit.sn."
)

add(
    [
        "Le DIT est-il présent sur les réseaux sociaux ?",
        "Sur quels réseaux sociaux peut-on suivre le DIT ?",
    ],
    "Le DIT est présent sur Facebook, X (anciennement Twitter), Instagram et LinkedIn, sous le nom "
    "Dakar Institute of Technology ou DIT Senegal."
)

# --- Reconnaissance et accréditation -----------------------------------
add(
    [
        "Le DIT est-il reconnu par l'État sénégalais ?",
        "Le Dakar Institute of Technology est-il accrédité ?",
        "Les diplômes du DIT sont-ils reconnus ?",
    ],
    "Oui. Le DIT est reconnu par le Ministère de la Formation professionnelle et technique du Sénégal "
    "depuis 2019, et par le Ministère de l'Enseignement supérieur, de la Recherche et de l'Innovation "
    "depuis 2021 (numéro RepSEN/Ensup-priv/AP/376-2021). Ses programmes de Licence Big Data et de "
    "Master en Intelligence Artificielle sont accrédités par l'ANAQ-SUP. L'école est aussi accréditée "
    "par le Fonds de financement de la formation professionnelle et technique (3FPT)."
)

add(
    [
        "Qu'est-ce que l'ANAQ-SUP a accordé au DIT ?",
        "Le DIT a-t-il récemment obtenu une nouvelle accréditation ?",
    ],
    "L'ANAQ-SUP, l'agence nationale d'assurance qualité placée sous la tutelle du ministère chargé de "
    "l'Enseignement supérieur, a accordé au DIT l'accréditation officielle de ses programmes de Licence "
    "en Big Data et de Master en Intelligence Artificielle, après un processus d'évaluation rigoureux."
)

add(
    [
        "Combien d'étudiants compte le DIT ?",
        "Le DIT est-il une grande école ?",
    ],
    "Le DIT accueille environ 200 étudiants, issus de plus de 17 nationalités différentes, ce qui en "
    "fait une école à taille humaine offrant un enseignement personnalisé."
)

# --- Programmes proposés (vue d'ensemble) --------------------------------
add(
    [
        "Quelles formations propose le DIT ?",
        "Quels sont les programmes disponibles au Dakar Institute of Technology ?",
        "Quelles filières peut-on suivre au DIT ?",
    ],
    "Le DIT propose deux licences (Informatique Big Data, et Business & Marketing Digital), deux "
    "masters (Intelligence Artificielle, et Finance Digitale), ainsi que plusieurs certifications "
    "courtes : TOEIC, Data Science Intensive, Data Manager, Python Basics et Fullstack Web."
)

# --- Licence Big Data ---------------------------------------------------
add(
    [
        "En quoi consiste la Licence Informatique Big Data du DIT ?",
        "Que peut-on apprendre en Licence Big Data au DIT ?",
        "Combien de temps dure la Licence Big Data ?",
    ],
    "La Licence Informatique Big Data du DIT est une formation de 3 ans, accréditée par l'ANAQ-SUP, "
    "axée sur la pratique. Elle prépare les étudiants à la gestion et au traitement de grands volumes "
    "de données, avec des compétences en programmation (Python, R, Scala), en stockage cloud et en "
    "analyse de données."
)

add(
    [
        "Que voit-on en première année de Licence Big Data ?",
        "Quel est le programme de Licence 1 en Big Data au DIT ?",
    ],
    "En Licence 1 Big Data, les étudiants étudient les systèmes Unix et l'installation de Linux, les "
    "bases du développement web (HTML, CSS, JS, hébergement, DNS), les langages Python et R, "
    "l'algorithmique, le stockage de données dans le cloud, une initiation à l'IoT avec Arduino, des "
    "outils d'IA en ligne, et réalisent un projet informatique."
)

add(
    [
        "Que voit-on en deuxième année de Licence Big Data ?",
        "Quel est le programme de Licence 2 en Big Data au DIT ?",
    ],
    "En Licence 2 Big Data, le programme couvre une initiation au Machine Learning, les conteneurs et "
    "la virtualisation, l'architecture Big Data, la visualisation de données, la collecte de données, "
    "l'ingénierie des données, la sécurité informatique (Infosec), et un projet informatique."
)

add(
    [
        "Que voit-on en troisième année de Licence Big Data ?",
        "Quel est le programme de Licence 3 en Big Data au DIT ?",
    ],
    "En Licence 3 Big Data, les étudiants abordent le management des données et le RGPD, un module "
    "start-up sous forme de jeu d'entreprise, une initiation à la programmation parallèle, le "
    "traitement du signal, l'art du pitch, la gestion de projet agile, la conception de chatbots, et "
    "effectuent un stage."
)

add(
    [
        "Quels métiers peut-on exercer après la Licence Big Data du DIT ?",
        "Quels débouchés offre la Licence Informatique Big Data ?",
    ],
    "La Licence Informatique Big Data du DIT prépare à des postes tels que Data Scientist junior, Data "
    "Engineer junior, Data Analyst junior, développeur Big Data, Data Quality Manager junior, "
    "statisticien, dataminer ou administrateur de bases de données."
)

add(
    [
        "Peut-on continuer ses études après la Licence Big Data du DIT ?",
        "Quelles poursuites d'études sont possibles après cette licence ?",
    ],
    "Après la Licence Big Data du DIT, les étudiants peuvent poursuivre vers un Master en Big Data, en "
    "Intelligence Artificielle, en Cybersécurité, en Robotique/Automatique, en Robotic Process "
    "Automation, ou en Réseaux et Systèmes."
)

add(
    [
        "Quelles sont les conditions d'admission en Licence Big Data au DIT ?",
        "Quels sont les prérequis pour intégrer la Licence Informatique Big Data ?",
    ],
    "Pour intégrer la Licence Informatique Big Data en première année, il faut être titulaire du "
    "baccalauréat, avoir une forte appétence pour l'informatique et être à l'aise en mathématiques. "
    "L'admission directe en Licence 2 ou 3 exige en plus d'avoir validé les crédits des années "
    "précédentes, de venir d'une filière scientifique ou technologique, d'avoir des bases en "
    "programmation, et de venir d'un établissement suivant le système LMD."
)

add(
    [
        "Comment se déroule le processus d'admission en Licence au DIT ?",
        "Faut-il se déplacer physiquement pour s'inscrire au DIT ?",
    ],
    "L'admission en Licence au DIT peut se faire entièrement en ligne, sans déplacement physique : elle "
    "comprend le dépôt du dossier de candidature, un entretien avec le Directeur des études, puis le "
    "paiement des frais d'inscription."
)

add(
    [
        "Quels documents faut-il fournir pour s'inscrire en Licence 1 au DIT ?",
        "Quelles pièces sont demandées pour l'inscription en première année ?",
    ],
    "Pour une inscription en Licence 1 au DIT, il faut fournir le relevé de notes du baccalauréat, le "
    "diplôme ou l'attestation de bac, ainsi qu'une carte d'identité ou un passeport."
)

add(
    [
        "Combien coûte la Licence 1 Big Data au DIT ?",
        "Quels sont les frais de scolarité de la première année de licence ?",
    ],
    "Pour la rentrée d'octobre, la Licence 1 Big Data au DIT coûte 1 100 000 FCFA au total, avec des "
    "frais d'inscription de 200 000 FCFA et des mensualités de 90 000 FCFA sur 10 mois."
)

add(
    [
        "Quand a lieu la prochaine rentrée en Licence au DIT ?",
        "À quelle date commence l'année scolaire en Licence Big Data ?",
    ],
    "La rentrée en Licence Informatique Big Data au DIT est prévue le 6 octobre 2026, avec une session "
    "alternative en janvier 2027."
)

# --- Master IA -----------------------------------------------------------
add(
    [
        "En quoi consiste le Master Intelligence Artificielle du DIT ?",
        "Que peut-on apprendre en Master IA au DIT ?",
        "Combien de temps dure le Master Intelligence Artificielle du DIT ?",
    ],
    "Le Master Intelligence Artificielle du DIT est un programme accrédité par l'ANAQ-SUP, étalé sur "
    "deux ans, disponible uniquement en cours du soir (en présentiel à Dakar ou en ligne). Il forme des "
    "experts capables de maîtriser l'ensemble du spectre de l'IA et de la data science."
)

add(
    [
        "Quelles spécialisations propose le Master IA du DIT en deuxième année ?",
        "Quels parcours peut-on choisir en M2 Intelligence Artificielle ?",
    ],
    "En deuxième année du Master Intelligence Artificielle, le DIT propose trois spécialisations : Data "
    "Scientist, Data Ingénieur, et Data Analyste."
)

add(
    [
        "Quelles matières sont enseignées dans le Master IA du DIT ?",
        "Quel est le programme du Master Intelligence Artificielle ?",
    ],
    "Le Master Intelligence Artificielle du DIT couvre notamment Python et R, les bases de données SQL "
    "et NoSQL, le DevOps, les outils statistiques pour la data science, la robotique et l'Internet des "
    "Objets, le traitement automatique du langage naturel, le traitement du signal, de la parole et de "
    "l'image, l'éthique de l'IA et la sécurité des données (RGPD), la blockchain, ainsi que "
    "l'entrepreneuriat, la gestion de projet, la communication et l'anglais."
)

add(
    [
        "Dans quels domaines d'application le Master IA du DIT propose-t-il des cas d'usage ?",
        "Quels secteurs peut-on approfondir dans le Master Intelligence Artificielle ?",
    ],
    "Les étudiants du Master IA du DIT peuvent approfondir des cas d'usage dans la reconnaissance "
    "d'image et la réalité virtuelle, l'Internet des Objets, le traitement automatique du langage "
    "naturel, le marketing et les ventes, la finance, ou la santé."
)

add(
    [
        "Quelles sont les conditions d'admission au Master IA du DIT ?",
        "Quel diplôme faut-il pour intégrer le Master Intelligence Artificielle ?",
    ],
    "Pour intégrer le Master Intelligence Artificielle du DIT, il faut être titulaire d'une licence "
    "scientifique (mathématiques, informatique, biologie, physique, etc.), avoir de bonnes aptitudes en "
    "mathématiques et une expérience dans au moins un langage de programmation."
)

add(
    [
        "Peut-on intégrer directement la deuxième année du Master IA au DIT ?",
        "L'admission directe en M2 Intelligence Artificielle est-elle possible ?",
    ],
    "Non. Seule l'admission en première année (M1) du Master Intelligence Artificielle est possible au "
    "DIT — l'école considère qu'apprendre l'intelligence artificielle est un programme ambitieux qu'il "
    "vaut mieux suivre en deux ans complets."
)

add(
    [
        "Combien coûte le Master Intelligence Artificielle au DIT ?",
        "Quels sont les frais du Master IA ?",
    ],
    "Les frais d'inscription au Master Intelligence Artificielle du DIT s'élèvent à 300 000 FCFA, avec "
    "une mensualité de 150 000 FCFA sur 10 mois."
)

add(
    [
        "Existe-t-il des bourses au DIT ?",
        "Le DIT propose-t-il un système de bourses pour le Master IA ?",
    ],
    "Non, le DIT ne dispose pas encore d'un système de bourses pour le Master Intelligence Artificielle "
    "à ce stade."
)

add(
    [
        "Peut-on suivre le Master IA du DIT à distance ?",
        "Le Master Intelligence Artificielle est-il accessible en ligne ?",
    ],
    "Oui. Les étudiants qui n'habitent pas à Dakar peuvent suivre le Master Intelligence Artificielle "
    "en ligne, en se connectant sur Zoom aux horaires des cours."
)

add(
    [
        "Quels sont les horaires des cours du Master IA au DIT ?",
        "À quelle heure ont lieu les cours du soir du Master Intelligence Artificielle ?",
    ],
    "Les cours du Master Intelligence Artificielle au DIT ont lieu du lundi au vendredi de 18h00 à "
    "20h30 (heure GMT), ainsi qu'un samedi par mois de 9h à 14h. Les examens ont généralement lieu le "
    "samedi."
)

add(
    [
        "Faut-il un ordinateur personnel pour suivre le Master IA au DIT ?",
        "Le DIT fournit-il un ordinateur aux étudiants du Master ?",
    ],
    "Non, les étudiants doivent posséder leur propre ordinateur portable (Mac ou PC) pour suivre les "
    "cours du Master Intelligence Artificielle."
)

add(
    [
        "Quand a lieu la prochaine rentrée du Master IA au DIT ?",
        "À quelle date commence le Master Intelligence Artificielle ?",
    ],
    "La rentrée du Master Intelligence Artificielle au DIT est prévue le 13 octobre 2026."
)

# --- Autres formations -----------------------------------------------
add(
    [
        "Qu'est-ce que la Licence Business et Marketing Digital du DIT ?",
        "En quoi consiste la formation en marketing digital au DIT ?",
    ],
    "La Licence Business et Marketing Digital du DIT est une formation professionnalisante centrée sur "
    "les leviers du marketing digital et les outils du commerce moderne : stratégies digitales, gestion "
    "de campagnes en ligne, analyse de données, et outils no-code et IA appliqués au marketing."
)

add(
    [
        "Qu'est-ce que le Master Finance Digitale du DIT ?",
        "En quoi consiste le Master en Finance Digitale ?",
    ],
    "Le Master Finance Digitale du DIT forme des experts à l'intersection de la finance et des "
    "technologies numériques, en couvrant la FinTech et le mobile money, la blockchain et les "
    "cryptomonnaies, l'intelligence artificielle appliquée à la finance, ainsi que la gestion des "
    "risques et la détection de fraudes. Il est ouvert aux titulaires d'une licence."
)

add(
    [
        "Quelles certifications courtes propose le DIT ?",
        "Quelles formations certifiantes peut-on suivre au DIT en dehors des licences et masters ?",
    ],
    "Le DIT propose plusieurs certifications courtes : la préparation au TOEIC, la formation Data "
    "Science Intensive (12 semaines en présentiel ou 24 semaines en ligne), la certification Data "
    "Manager (12 semaines), la certification Python Basics (3 semaines) et la certification Fullstack "
    "Web (19 semaines)."
)

add(
    [
        "En combien de temps peut-on obtenir la certification Python Basics au DIT ?",
        "Combien de temps dure la formation Python Basics ?",
    ],
    "La certification Python Basics du DIT se prépare en 3 semaines, en cours du soir à distance, et "
    "couvre la syntaxe de Python, la manipulation de données et les bases de la programmation orientée "
    "objet."
)

add(
    [
        "En combien de temps peut-on obtenir la certification Fullstack Web au DIT ?",
        "Combien de temps dure la formation Fullstack Web ?",
    ],
    "La certification Fullstack Web du DIT se prépare en 19 semaines, en cours du soir à distance, et "
    "couvre le développement front-end et back-end, du HTML/CSS à JavaScript, avec des frameworks comme "
    "React et Angular."
)

# --- Équipe --------------------------------------------------------------
add(
    [
        "Qui est le Directeur des études du DIT ?",
        "Qui occupe le poste de Chief Academic Officer au DIT ?",
    ],
    "Le Directeur des études (Chief Academic Officer) du DIT est le Dr Seydou Nourou Sylla."
)

add(
    [
        "Qui compose l'équipe de direction du DIT ?",
        "Quels sont les principaux responsables du DIT ?",
    ],
    "L'équipe de direction du DIT comprend le Dr Nicolas Poussielgue (Directeur général), le Dr Seydou "
    "Nourou Sylla (Directeur des études), Suzanne Kh. Sagne (Responsable Commerciale), Dominique Ndour "
    "(Responsable Pédagogique du cycle Licence), Ndéye Déguène Fall (Responsable Administrative et "
    "Financière), et Boubacar Diallo (Responsable Qualité)."
)

# --- International ---------------------------------------------------
add(
    [
        "Le DIT a-t-il des partenariats internationaux ?",
        "Avec quelles universités étrangères le DIT collabore-t-il ?",
        "Peut-on faire un échange international depuis le DIT ?",
    ],
    "Oui, le DIT entretient des partenariats internationaux depuis 2022 avec plusieurs établissements : "
    "EPITA, ESIEA et ESTIA en France, l'université Gaziosmanpaşa en Turquie, l'université de Nanjing "
    "des sciences et technologies en Chine, et Alma College aux États-Unis. Ces partenariats permettent "
    "des semestres d'échange, des doubles diplômes et des expériences d'immersion internationale."
)

# --- Vie étudiante ---------------------------------------------------
add(
    [
        "Qu'est-ce que STEM Pour Elles au DIT ?",
        "Le DIT organise-t-il des événements pour promouvoir les femmes dans la tech ?",
    ],
    "STEM Pour Elles est un événement organisé par le DIT pour promouvoir la place des femmes dans les "
    "sciences, la technologie, l'ingénierie et les mathématiques. La troisième édition a eu lieu le "
    "25 avril 2026."
)

add(
    [
        "Le DIT organise-t-il une cérémonie de remise de diplômes ?",
        "Quand a eu lieu la dernière cérémonie de graduation du DIT ?",
    ],
    "Oui, le DIT organise une cérémonie de graduation pour ses diplômés. La promotion 2025-2026 de "
    "Licence Big Data et de Master Intelligence Artificielle a été célébrée le 20 juin 2026."
)

# --- Master IA : compléments (débouchés, poursuite d'études, corps enseignant) ---
add(
    [
        "Quels métiers peut-on exercer après le Master IA du DIT ?",
        "Quels débouchés offre le Master Intelligence Artificielle ?",
    ],
    "Le Master Intelligence Artificielle du DIT prépare à des postes tels que Data Analyst, Data "
    "Scientist, Data Ingénieur, Chief Data Officer, Data Protection Officer, Architecte Big Data, "
    "Consultant Big Data/IA, ou encore Entrepreneur dans l'IA."
)

add(
    [
        "Peut-on poursuivre en doctorat après le Master IA du DIT ?",
        "Quelles poursuites d'études sont possibles après le Master Intelligence Artificielle ?",
    ],
    "Oui, les diplômés du Master Intelligence Artificielle du DIT peuvent poursuivre en doctorat en "
    "informatique, avec des mentions possibles en Intelligence Artificielle, Machine Learning, Deep "
    "Learning, Reinforcement Learning ou Big Data."
)

add(
    [
        "Qui enseigne au Master IA du DIT ?",
        "Quel est le profil des professeurs du Master Intelligence Artificielle ?",
    ],
    "Les professeurs du Master Intelligence Artificielle du DIT sont africains et européens, choisis "
    "parmi des spécialistes de leur domaine. Des séminaires réguliers sont aussi organisés avec des "
    "experts locaux et internationaux pour donner aux étudiants une vue globale des enjeux de l'IA."
)

# --- Licence Business et Marketing Digital : programme détaillé par semestre ---
add(
    [
        "Quels cours sont enseignés au premier semestre de la Licence Business et Marketing Digital ?",
        "Quel est le programme du semestre 1 de la Licence Business et Marketing Digital ?",
    ],
    "Au premier semestre de la Licence Business et Marketing Digital du DIT, les étudiants suivent : "
    "Introduction générale à l'économie, Droit des sociétés, Anglais 1, Bureautique, Technique "
    "d'expression et de communication 1, Introduction au Marketing Digital, Marketing Stratégique, "
    "Communication professionnelle, Outils CRM, et Collecte de données en ligne."
)

add(
    [
        "Quels cours sont enseignés au deuxième semestre de la Licence Business et Marketing Digital ?",
        "Quel est le programme du semestre 2 de la Licence Business et Marketing Digital ?",
    ],
    "Au deuxième semestre de la Licence Business et Marketing Digital du DIT, le programme comprend : "
    "Introduction à la fiscalité, Introduction à la Comptabilité Générale, Outils Statistiques, Anglais "
    "2, Bureautique dans le Cloud, Techniques d'Expression et de Communication 2, Marketing "
    "Opérationnel, Introduction au Marketing Digital (approfondissement), Infographie, Stratégie de "
    "contenu en marketing Digital, et Outils Web."
)

add(
    [
        "Quels cours sont enseignés en troisième semestre (Licence 2) de la Licence Business et Marketing Digital ?",
        "Quel est le programme du semestre 3 de la Licence Business et Marketing Digital ?",
    ],
    "Au troisième semestre (début de Licence 2) de la Licence Business et Marketing Digital du DIT, on "
    "trouve : Outils No Code et RPA 1, Base de données, Anglais 3, Gestion de ressources humaines, "
    "Gestion digitale de la relation commerciale, Introduction à l'intelligence artificielle, "
    "E-Commerce et E-Logistique, Référencement Web et SEO, et Community Management et création de "
    "contenus."
)

add(
    [
        "Quels cours sont enseignés en quatrième semestre (Licence 2) de la Licence Business et Marketing Digital ?",
        "Quel est le programme du semestre 4 de la Licence Business et Marketing Digital ?",
    ],
    "Au quatrième semestre de la Licence Business et Marketing Digital du DIT, le programme comprend : "
    "Droit commercial, Anglais 4, Management interculturel, Études et Analyse de Marché, "
    "Planification stratégique et Gestion d'Entreprise, Stratégie Digitale et Gestion de campagne, "
    "Introduction au Growth Hacking, Outils No Code et RPA 2, et Introduction ERP 1."
)

add(
    [
        "Quels cours sont enseignés en cinquième semestre (Licence 3) de la Licence Business et Marketing Digital ?",
        "Quel est le programme du semestre 5 de la Licence Business et Marketing Digital ?",
    ],
    "Au cinquième semestre (début de Licence 3) de la Licence Business et Marketing Digital du DIT, on "
    "trouve : Rentabilité financière, Planification Financière et gestion des risques, Anglais 5 "
    "(préparation au TOEIC, objectif 700 points), Gestion de projet agile, Méthodologie de rédaction "
    "de mémoire, Culture et Consommation digitale, Motion Design, Stratégie des KPI et Dashboards, et "
    "Outils IA en ligne."
)

add(
    [
        "Quels cours sont enseignés en sixième semestre (Licence 3) de la Licence Business et Marketing Digital ?",
        "Quel est le programme du semestre 6 de la Licence Business et Marketing Digital ?",
    ],
    "Au sixième et dernier semestre de la Licence Business et Marketing Digital du DIT, le programme "
    "comprend : Droit du numérique et RGPD, Gestion de crise en ligne, L'art du Pitch, Réalité "
    "virtuelle et augmentée, Introduction aux Chatbots, et Jeu d'entreprise (Business Game)."
)

# --- Licence Business : quelques cours notables, en détail ---
add(
    [
        "Que couvre le cours d'introduction à l'intelligence artificielle dans la Licence Business et Marketing Digital ?",
        "Les étudiants de la Licence Business apprennent-ils l'IA ?",
    ],
    "Oui. En Licence 2, un cours d'introduction à l'intelligence artificielle donne une vue d'ensemble "
    "des concepts fondamentaux de l'IA (agents intelligents, recherche de solutions, apprentissage "
    "automatique), explore ses applications dans les jeux, la robotique, la vision par ordinateur et "
    "le langage naturel, et aborde les enjeux éthiques liés à l'équité et à la transparence des "
    "systèmes intelligents."
)

add(
    [
        "Que couvre le cours sur les chatbots dans la Licence Business et Marketing Digital ?",
        "Apprend-on à créer des chatbots au DIT en Licence Business ?",
    ],
    "Oui, en Licence 3, un cours dédié explore l'histoire et le fonctionnement des chatbots, les "
    "principes du traitement automatique du langage naturel, et l'utilisation d'outils no-code comme "
    "Botpress, ManyChat ou Chatfuel pour créer des chatbots sans programmation, avec une introduction "
    "aux LLM génératifs comme ChatGPT."
)

add(
    [
        "Que couvre le cours sur le Growth Hacking dans la Licence Business et Marketing Digital ?",
        "Qu'apprend-on en Growth Hacking au DIT ?",
    ],
    "Le cours de Growth Hacking, en Licence 2, initie les étudiants aux stratégies agiles "
    "d'accélération de croissance via le cadre AARRR (Acquisition, Activation, Rétention, "
    "Recommandation, Revenu), avec des outils comme Hotjar pour l'analyse comportementale et Expandi.io "
    "pour l'automatisation sur LinkedIn."
)

add(
    [
        "Le RGPD est-il enseigné à la Licence Business et Marketing Digital du DIT ?",
        "Que couvre le cours sur le droit du numérique et le RGPD ?",
    ],
    "Oui, en Licence 3, un cours de droit du numérique et RGPD couvre les implications juridiques du "
    "marketing digital, les mesures de sécurité pour protéger les données personnelles, et les "
    "obligations légales de consentement et de respect de la vie privée."
)

add(
    [
        "Que couvre le cours sur les outils IA en ligne de la Licence Business et Marketing Digital ?",
        "Quels outils d'IA générative sont enseignés en Licence 3 Business ?",
    ],
    "Ce cours de Licence 3 initie les étudiants à l'utilisation stratégique d'outils d'IA générative "
    "dans le cadre universitaire : ChatGPT pour la rédaction assistée, Leo AI pour les plans et fiches "
    "de révision, NotebookLM pour la prise de notes, et Quizlet pour les flashcards, avec un regard "
    "critique sur leurs limites et enjeux éthiques."
)

add(
    [
        "Y a-t-il un cours de préparation au TOEIC dans la Licence Business et Marketing Digital ?",
        "Comment se déroule l'apprentissage de l'anglais dans cette licence ?",
    ],
    "L'anglais est enseigné sur cinq semestres consécutifs (Anglais 1 à 5) dans la Licence Business et "
    "Marketing Digital du DIT, avec une progression du niveau ESL débutant jusqu'à la préparation "
    "complète du TOEIC listening and reading, avec un objectif minimal de 700 points (niveau B2 du "
    "CECRL) au cinquième semestre."
)

add(
    [
        "Y a-t-il un stage ou un mémoire à la fin de la Licence Business et Marketing Digital ?",
        "Comment se termine la Licence Business et Marketing Digital du DIT ?",
    ],
    "La Licence Business et Marketing Digital comprend un cours dédié de méthodologie de rédaction de "
    "mémoire en Licence 3, couvrant la définition de la problématique, la recherche de sources "
    "(Google Scholar, Zotero, Mendeley), la collecte de données qualitatives et quantitatives, et la "
    "rédaction dans les règles de l'art académique."
)


add(
    [
        "Comment puis-je payer mes frais de scolarité au DIT ?",
        "Quels sont les moyens de paiement acceptés par le DIT ?",
        "Comment régler ma scolarité au Dakar Institute of Technology ?",
    ],
    "Le DIT accepte plusieurs modes de paiement : virement bancaire (CBAO ou UBA), Orange Money, Wave, "
    "PayPal, et paiement par carte bancaire (Visa ou Mastercard) via un lien sécurisé. Après tout "
    "paiement, il faut envoyer un message sur WhatsApp au +221 77 598 51 51 ou un e-mail à "
    "compta@dit.sn pour obtenir un reçu."
)

add(
    [
        "Peut-on payer le DIT par virement bancaire ?",
        "Quelles sont les coordonnées bancaires du DIT ?",
        "Sur quel compte bancaire faut-il virer les frais de scolarité du DIT ?",
    ],
    "Le DIT dispose de deux comptes bancaires pour les virements, à l'agence de Dakar : un compte à la "
    "CBAO Groupe Attijariwafa Bank (IBAN SN08 SN01 2012 3203 6193 8286 0143, code SWIFT CBAOSNDA), et "
    "un compte à l'UBA (IBAN SN40 SN15 3013 0530 5090 0029 7385, code SWIFT UNAFSNDA). Il est important "
    "d'indiquer le numéro de facture dans le libellé du virement."
)

add(
    [
        "Peut-on payer le DIT avec Orange Money ?",
        "Comment utiliser Orange Money pour payer le DIT ?",
    ],
    "Oui, le DIT accepte Orange Money via le code marchand 413515 (composer #144#5*413515*MONTANT*"
    "CODESECRET# puis appeler, uniquement depuis le Sénégal), ou en envoyant le paiement au numéro "
    "+221 77 381 82 82, qui fonctionne depuis le Sénégal et les autres pays disposant d'Orange Money."
)

add(
    [
        "Peut-on payer le DIT avec Wave ?",
        "Comment utiliser Wave pour régler ses frais au DIT ?",
    ],
    "Oui, depuis le Sénégal, le paiement via Wave se fait directement depuis l'application mobile Wave "
    "en scannant le QR code fourni par le DIT ou via un lien de paiement dédié. Depuis un autre pays, "
    "il faut faire un transfert vers le numéro +221 77 381 82 82."
)

add(
    [
        "Peut-on payer le DIT par PayPal ?",
        "Comment régler ses frais de scolarité via PayPal au DIT ?",
    ],
    "Oui, le DIT accepte PayPal à l'adresse paypal@dit.sn. Il est recommandé de choisir le mode « envoi "
    "entre proches » pour éviter des frais, ou d'ajouter 5 % du montant si l'on choisit le mode "
    "« paiement de biens et services », afin de couvrir les frais PayPal."
)

add(
    [
        "Peut-on payer le DIT par carte bancaire ?",
        "Le DIT accepte-t-il les paiements par Visa ou Mastercard ?",
    ],
    "Oui, le DIT accepte les paiements par carte bancaire (Visa et Mastercard) via un lien de paiement "
    "sécurisé fourni par l'école."
)



# CORRECTIF -- reproduit exactement le format de scripts/prepare_dataset.py
# (format_alpaca_example), utilisé pour TOUT le corpus général sur lequel
# les checkpoints existants ont été entraînés : le champ "prompt" doit
# contenir le gabarit "### Instruction:\n...\n\n### Réponse:\n" DIRECTEMENT
# (build_example() ne l'ajoute PAS lui-même), et la réponse doit commencer
# par un espace. Sans ce correctif, un fine-tuning à partir d'un checkpoint
# général verrait un format différent de celui qu'il a appris -- un vrai
# décalage de distribution, indépendant du contenu factuel.
for item in DATA:
    item["prompt"] = f"### Instruction:\n{item['prompt']}\n\n### Réponse:\n"
    item["response"] = " " + item["response"]

random.shuffle(DATA)

import os

os.makedirs("outputs/data_dit", exist_ok=True)

with open("outputs/data_dit/dit_dataset.json", "w", encoding="utf-8") as f:
    json.dump(DATA, f, ensure_ascii=False, indent=2)

print(f"{len(DATA)} exemples générés -> dit_dataset.json")

# Répartition train/val (90/10, comme le reste du projet)
n_val = max(5, len(DATA) // 10)
val = DATA[:n_val]
train = DATA[n_val:]

with open("outputs/data_dit/dit_train.json", "w", encoding="utf-8") as f:
    json.dump(train, f, ensure_ascii=False, indent=2)
with open("outputs/data_dit/dit_val.json", "w", encoding="utf-8") as f:
    json.dump(val, f, ensure_ascii=False, indent=2)

print(f"Train: {len(train)} exemples -> dit_train.json")
print(f"Val:   {len(val)} exemples -> dit_val.json")
